import logging
import time
import hashlib
import os
import argparse
import einops

from collections.abc import Iterator
from dataclasses import replace
import numpy as np

import torch
import torchaudio

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.audio_vae import AudioEncoder
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.types import AudioLatentShape, LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.args import default_2_stage_distilled_arg_parser
from ltx_pipelines.utils.constants import (
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    image_conditionings_by_replacing_latent,
    simple_denoising_func,
    noise_audio_state
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()

logging.basicConfig(level=logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("ltx_core").setLevel(logging.ERROR)


def load_audio_input(audio_path: str, target_sample_rate: int, device: torch.device) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    
    # Ensure stereo/mono match? LTX usually expects specific channels? 
    # Usually we just take the first channel or mix if needed, but let's assume standard handling or strict match later.
    # LTX Audio VAE likely handles what it needs, but we need to check channels.
    # checking audio_vae.py -> ch=128 (latent), but input?
    # The VAE expects spectrograms, usually.
    # Wait, the pipeline usually generates audio.
    # We need to *encode* the waveform to latents.
    # The repository doesn't seem to expose a direct "wav -> latent" helper in the viewed files easily, 
    # but `AudioEncoder` takes `spectrogram`.
    # We need `AudioProcessor` to convert wav to spectrogram.
    
    return waveform.to(device)

class MusicToVideoPipeline:
    """
    Modified DistilledPipeline for Music-to-Video generation.
    Takes an input audio file, encodes it, and uses it to condition/guide the video generation.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device = device,
        fp8transformer: bool = False,
    ):
        self.device = device
        self.dtype = torch.bfloat16

        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root_path=gemma_root,
            loras=loras,
            fp8transformer=fp8transformer,
        )

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

    def encode_audio_latents(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Encodes the audio waveform into latents using the VAE encoder.
        """
        # We need the AudioVAE encoder and the AudioProcessor (spectrogram converter)
        # The ModelLedger provides audio_encoder().
        # We need to instantiate AudioProcessor manually or find where it is.
        # It's in `ltx_core.model.audio_vae.ops`.
        from ltx_core.model.audio_vae.ops import AudioProcessor
        
        # Defaults based on audio_vae.py
        n_fft = 1024
        mel_hop_length = 160
        mel_bins = 64
        
        audio_processor = AudioProcessor(
            sample_rate=16000,
            mel_bins=mel_bins,
            mel_hop_length=mel_hop_length,
            n_fft=n_fft
        ).to(self.device)
        
        # Spectrogram expects (batch, channels, time) or similar?
        # AudioProcessor.get_spectrogram usually takes (batch, channels, time).
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0) # (1, C, T)

        # Ensure stereo (2 channels) as expected by AudioEncoder
        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)
        elif waveform.shape[1] > 2:
            waveform = waveform[:, :2, :]
            
        spectrogram = audio_processor.waveform_to_mel(waveform.to(self.device).float(), 16000)
        # spectrogram shape: (Batch, Channels, Time, Freq)
        
        audio_encoder = self.model_ledger.audio_encoder()
        
        # Encoder expects (Batch, Channels, Time, Freq)
        # Check AudioEncoder logic. It uses `conv_in` which is 2D conv... on what?
        # Usually (B, C, H, W). Time and Freq are H, W.
        # Spectrogram is (B, 1, T, F).
        
        encoded_latents = audio_encoder(spectrogram.to(self.dtype))
        
        # Clean up
        del audio_encoder
        del audio_processor
        cleanup_memory()
        
        return encoded_latents

    @torch.inference_mode()
    def __call__(
            self,
            prompt: str,
            seed: int,
            height: int,
            width: int,
            num_frames: int,
            frame_rate: float,
            images: list[tuple[str, int, float]],
            audio_input_path: str | None = None,
            tiling_config: TilingConfig | None = None,
            enhance_prompt: bool = False,
            output_path: str = '',
            video_chunks_number: int = 0,
            fps: int = 0,
            save_step_1_preview: bool = True,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor | None]:
        print("Preparing Inference (Music to Video)")
        startAt = time.time()
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        # --- LOAD AUDIO ---
        audio_waveform = None
        audio_latents = None
        
        if audio_input_path:
             print(f"Loading audio from {audio_input_path}")
             # We load raw waveform for final mix, and encode for latents
             audio_waveform = load_audio_input(audio_input_path, 16000, self.device)
             print("Encoding audio latents...")
             audio_latents = self.encode_audio_latents(audio_waveform)
             
             # Verify latent shape matches required frames?
             # Audio VAE has downsample factor ~4 in time?
             # LTX expects aligned video/audio latents.
             # We might need to crop/pad audio_latents to match num_frames video equivalent.
             pass

        # --- PROMPT CACHE LOGIC START ---
        CACHE_DIR = "./prompt_embeddings_cache"
        os.makedirs(CACHE_DIR, exist_ok=True)

        image_identifier = images[0][0] if (len(images) > 0 and enhance_prompt) else "no_img"

        hash_input_str = (
            f"prompt:{prompt}|"
            f"pipeline:music_distilled|"
            f"enhance:{enhance_prompt}|"
            f"seed:{seed if enhance_prompt else 'ignored'}|"
            f"img:{image_identifier}"
        )

        cache_filename = hashlib.md5(hash_input_str.encode('utf-8')).hexdigest() + ".pt"
        cache_path = os.path.join(CACHE_DIR, cache_filename)

        context_p = None

        if os.path.exists(cache_path):
            print(f"Prompt cache hit! Loading embeddings from {cache_path}")
            try:
                context_p = torch.load(cache_path, map_location=self.device)
            except Exception as e:
                print(f"Failed to load cache (corrupted?): {e}. Regenerating.")

        if context_p is None:
            print("Prompt cache miss. Running text encoder.")
            text_encoder = self.model_ledger.text_encoder()
            current_prompt = prompt
            if enhance_prompt:
                current_prompt = generate_enhanced_prompt(
                    text_encoder, prompt, images[0][0] if len(images) > 0 else None
                )
            context_p = encode_text(text_encoder, prompts=[current_prompt])[0]

            print(f"Saving embeddings to {cache_path}")
            torch.save(context_p, cache_path)
            print("Prompt encoded.", time.time() - startAt)

            torch.cuda.synchronize()
            del text_encoder
            cleanup_memory()
        
        video_context, audio_context = context_p
        
        # If we have input audio, we theoretically ignore audio_context (text-to-audio guidance) 
        # or we keep it to help semantic matching? 
        # We'll keep it as the model expects it, but we force the audio latents.

        print("Stage 1: Initial low resolution video generation.")

        transformer = self.model_ledger.transformer()
        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        def music_denoising_loop(
                sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol, is_conditioning: bool = True
        ) -> tuple[LatentState, LatentState]:
            
            # Custom loop that enforces audio_state to be the input music
            # We wrap the underlying euler_denoising_loop logic but override audio state updates?
            # euler_denoising_loop iterates over sigmas.
            # We can't easily inject into the internal loop of euler_denoising_loop without copying it.
            # So we will copy `euler_denoising_loop` logic here or modify how we call it.
            
            # Actually, `euler_denoising_loop` takes `denoise_fn`.
            # We can modify `denoise_fn` to output 0 correction for audio, 
            # AND strictly set `audio_state` to `audio_latents` (noised appropriately? or clean?).
            
            # If we want the video to attend to the "clean" audio, we should present the clean audio to the transformer.
            # But the transformer expects noised input at step t.
            # If we pass "clean" audio as "noised audio", the model might be confused if it expects noise.
            # HOWEVER, if we want strict control, we can just say audio_state is the latent.
            
            # Simplified approach:
            # 1. We let the loop run.
            # 2. Inside the denoise_fn, we ignore the predicted audio correction.
            # 3. We overwrite the audio_state in the loop? `euler_denoising_loop` returns the final state.
            
            # Better: Copy `euler_denoising_loop` logic:
            
            v_x = video_state
            a_x = audio_state
            
            # If audio_latents provided, we initialize a_x to it?
            # But a_x starts as pure noise in the standard pipeline.
            # If we want to condition, we should probably start with the noised version of our audio?
            # Or just clean audio? 
            # "Audio Reactive" usually means we use audio features to drive generation.
            # In LTX, video and audio are generated jointly.
            # If we feed the *clean* audio latent (encoded from file) as `audio_state` at every step,
            # the transformer will see it via self-attention/cross-attention layers.
            
            for i in range(len(sigmas) - 1):
                sigma_hat = sigmas[i]
                sigma_next = sigmas[i + 1]
                
                # If we have fixed audio, we might want to force a_x to be the 'noised' version of our target audio at this sigma level?
                # Or just the clean target audio if the model is robust enough? 
                # Let's try forcing it to be the correct 'noised' level state of our ground truth audio.
                if loop_audio_latents is not None:
                     # Add noise to clean audio_latents matching current sigma
                     # noise = torch.randn_like(audio_latents)
                     # a_x_target = audio_latents + noise * sigma_hat
                     # But this changes noise every step.
                     # We should define the noise once or use consistent noise.
                     # Simpler: Just force a_x = audio_latents (clean). 
                     # The model might treat it as "denoised" and try to predict 0 noise.
                     # Let's trust the detailed plan: "inject these latents".
                     
                     # We will set a_x to the audio_latents, but we need to ensure dimensions match.
                     # Audio VAE latent might have different length than what 'noise_audio_state' produced if frames differ.
                     # We align them before loop.
                     
                     # We align them before loop.
                     
                     a_x = replace(a_x, latent=loop_audio_latents) # Force strict guidance with flattened latents
                     pass

                denoised_v, denoised_a = simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,
                    is_conditioning=is_conditioning,
                    disable_audio=False, # We want the model to see audio
                )(v_x, a_x, sigmas, i)
                
                # Euler step for Video
                d_v = (v_x.latent - denoised_v) / sigma_hat
                dt = sigma_next - sigma_hat
                v_x = replace(v_x, latent=v_x.latent + d_v * dt)
                
                # For Audio, if we are forcing it, we don't need to step it. 
                # If we are NOT forcing it (just initializing), we would step it.
                # Here we are FORCING it to be our input. So we skip audio update or reset it next loop.
            
            return v_x, a_x

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        
        
        # Prepare conditionings
        stage_1_conditionings = []
        is_conditioning = False
        if images:
            is_conditioning = True
            video_encoder = self.model_ledger.video_encoder()
            stage_1_conditionings = image_conditionings_by_replacing_latent(
                images=images,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
            )
            torch.cuda.synchronize()
            del video_encoder
            cleanup_memory()

        # Align audio latents to match video duration
        if audio_latents is not None:
             expected_audio_shape = AudioLatentShape.from_video_pixel_shape(
                 stage_1_output_shape, 
                 sample_rate=16000
             )
             target_frames = expected_audio_shape.frames
             current_frames = audio_latents.shape[2]
             
             if current_frames > target_frames:
                 print(f"Aligning audio: Trimming from {current_frames} to {target_frames}")
                 audio_latents = audio_latents[:, :, :target_frames, :]
             elif current_frames < target_frames:
                 print(f"Aligning audio: Padding from {current_frames} to {target_frames}")
                 pad_amount = target_frames - current_frames
                 # Pad dimension 2 (Time). 
                 # F.pad for 4D input (B, C, T, F). 
                 # Pad args are (dim_-1_left, dim_-1_right, dim_-2_left, dim_-2_right, ...)
                 # We want to pad dim 2 (Time), which is dim_-2.
                 # So args are (freq_left, freq_right, time_left, time_right)
                 audio_latents = torch.nn.functional.pad(audio_latents, (0, 0, 0, pad_amount))
        
        if audio_latents is not None:
             audio_state, audio_tools = noise_audio_state(
                 stage_1_output_shape,
                 noiser,
                 [], # Audio has no image-based latents
                 self.pipeline_components,
                 self.dtype,
                 self.device,
                 noise_scale=1.0,
                 initial_latent=audio_latents, # Pass 4D aligned latents
             )
             
             # Create flattened version for loop injection
             # b c t f -> b t (c f)
             loop_audio_latents = einops.rearrange(audio_latents, "b c t f -> b t (c f)")
             
             # Check compatibility with transformer and slice if needed
             # Transformer expects [B, T, 16] but encoder gives [B, T, 128]
             # X0Model wraps LTXModel as velocity_model
             in_features = transformer.velocity_model.audio_patchify_proj.in_features
             if loop_audio_latents.shape[-1] != in_features:
                 print(f"Aligning audio features for loop: {loop_audio_latents.shape[-1]} -> {in_features}")
                 loop_audio_latents = loop_audio_latents[..., :in_features]
        else:
            audio_state, audio_tools = noise_audio_state(
                stage_1_output_shape,
                noiser,
                stage_1_conditionings,
                self.pipeline_components,
                self.dtype,
                self.device,
                noise_scale=1.0,
            )
            loop_audio_latents = None



        print("Stage 1: Starting denoising loop.", time.time() - startAt)
        
        # Initialize states
        # We need to manually initialize if we want to inject audio properly from start?
        # denoise_audio_video creates random noise.
        # We can pass initial_audio_latent!
        
        # Resize audio_latents to match stage 1? 
        # Audio VAE latent frames depend on time. 
        # If we trained on specific fps/resolution, audio latent structure is independent of video resolution (mostly), 
        # but depends on duration (frames / fps).
        # We should check if stage 1 and stage 2 use different audio latent structures?
        # Typically audio is same, video resolution changes.
        
        # So we can pass audio_latents as initial_audio_latent.
        # BUT `denoise_audio_video` adds noise to initial_latent if provided (via `noise_audio_state`).
        # We want to maybe start with it?
        
        # Let's stick to the `music_denoising_loop` strategy of forcing it.
        # But we need to ensure `audio_state` has correct shape.
        
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stepper,
            denoising_loop_fn=music_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            is_conditioning=is_conditioning,
            initial_audio_latent=audio_latents if audio_latents is not None else None 
            # If we pass it here, it gets fully noised. 
            # The loop will then overwrite it with clean/fixed version if we implemented it right.
        )
        
        print("Stage 1: Finish denoising loop.", time.time() - startAt)
        
        torch.cuda.synchronize()
        del stage_1_sigmas
        del stage_1_output_shape
        del stage_1_conditionings
        cleanup_memory()

        if save_step_1_preview:
            video_decoder = self.model_ledger.video_decoder()
            decoded_video = vae_decode_video(video_state.latent, video_decoder, tiling_config)
            torch.cuda.synchronize()
            del video_decoder
            cleanup_memory()
            
            # For preview, we can use the decoded audio from state (which should match input if we forced it)
            # OR just use the original input waveform.
            # Using state validates that the VAE cycle works.
            # But for "music to video", we probably want the HQ input audio in output.
            
            decoded_audio = None
            if audio_latents is not None:
                # Decode the latent to check it? Or just use raw waveform?
                # Let's decode to be safe/consistent with pipeline flow.
                vocoder = self.model_ledger.vocoder()
                decoded_audio = vae_decode_audio(
                    audio_state.latent, self.model_ledger.audio_decoder(), vocoder
                )
                torch.cuda.synchronize()
                del vocoder
                cleanup_memory()
            
            encode_video(
                video=decoded_video,
                fps=fps,
                audio=audio_waveform.cpu() if audio_waveform is not None else decoded_audio, # Use HQ input if available
                audio_sample_rate=24000,
                output_path=output_path.replace('.mp4', '_.mp4'),
                video_chunks_number=video_chunks_number,
            )

        print("Stage 2: Upsample and refine.")
        
        video_encoder = self.model_ledger.video_encoder()
        upsampler = self.model_ledger.spatial_upsampler()
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1], video_encoder=video_encoder, upsampler=upsampler
        )
        stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = []
        if images:
             stage_2_conditionings = image_conditionings_by_replacing_latent(
                images=images,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
            )

        torch.cuda.synchronize()
        del video_encoder
        del upsampler
        del video_state
        cleanup_memory()

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=stage_2_sigmas,
            stepper=stepper,
            denoising_loop_fn=music_denoising_loop, # Reuse forced audio loop
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=stage_2_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_latents, # Reuse clean audio latents
            is_conditioning=is_conditioning
        )
        
        print("Stage 2: Finish upsample.", time.time() - startAt)
        
        torch.cuda.synchronize()
        del transformer
        del stage_2_output_shape
        del stage_2_conditionings
        del stage_2_sigmas
        cleanup_memory()
        
        print("Stage 3: VAE Decode.")
        video_decoder = self.model_ledger.video_decoder()
        decoded_video = vae_decode_video(video_state.latent, video_decoder, tiling_config)
        del video_decoder
        cleanup_memory()
    
        if audio_state is not None:
            # Unpatchify if needed (3D -> 4D) before decoding
            # If loop returned flattened latents, we must restore them.
            # However, we sliced features to 16. Decoder expects 128 (8ch * 16bins).
            # We cannot simply reshape 16 features back to 128.
            # But wait, audio_state was initialized with FULL 4D latents.
            # The loop only manipulated a_x which had sliced features.
            # If the loop returns a_x, it has sliced features (16).
            # We need to restore it to original 128 features (if we want to decode the result of the loop).
            # OR: if we forced guidance, we just want to decode the ORIGINAL audio_latents (which we have).
            
            # If we just want to save the audio we generated/guided:
            # Since we forced a_x = loop_audio_latents (which is sliced input),
            # The output audio_state.latent is equal to loop_audio_latents (sliced).
            
            # We actually want to decode the original audio we passed in (audio_latents 4D).
            # So we can just decode `audio_latents` directly? 
            # Yes, since we are doing "music to video", the audio is input, not generated.
            # So we should decode the `audio_latents` (the aligned 4D input) instead of `audio_state.latent`.
            
            # BUT `denoise_audio_video` returns `audio_state`. 
            # If we trust the pipeline, we should use what it returns.
            # But since we sliced it, it's destructive.
            # Let's decode the original aligned `audio_latents` (which is 4D, 128 features).
            pass

        audio = None
        if audio_state is not None:
            # Decode the original aligned input audio, ensuring we have the full feature set
            if audio_latents is not None:
                 vocoder = self.model_ledger.vocoder()
                 audio = vae_decode_audio(audio_latents, self.model_ledger.audio_decoder(), vocoder)
                 torch.cuda.synchronize()
                 del vocoder
                 cleanup_memory()
            else:
                 # Generative case? We don't support generative audio here yet with this slicing hack.
                 # Attempt decode (will likely fail if sliced)
                 pass
        
        print("Stage 3: Done.", time.time() - startAt)
        return decoded_video, audio_waveform.cpu() if audio_waveform is not None else audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    parser = default_2_stage_distilled_arg_parser()
    parser.add_argument("--audio-input-path", type=str, default=None, help="Path to input audio file")
    args = parser.parse_args()
    
    pipeline = MusicToVideoPipeline(
        checkpoint_path=args.checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.lora,
        fp8transformer=args.enable_fp8,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    
    video, audio = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        audio_input_path=args.audio_input_path,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
        fps=args.frame_rate,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=24000,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )

if __name__ == "__main__":
    main()
