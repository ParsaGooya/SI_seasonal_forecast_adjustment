multi-channel partial convs as before, no skip connections, layernormalization, biases turned True, Obs mask used for prior model, skip VAE

Runs with * in their name have smoothed land masked loss weights.

NPSproj made smaller in longitude.

V0417 introduces noise to decoder and trains with CRPS.

V0707 increases number of noise injections.  Also clips the deterministic initial guess to (0,1) for saved unet. Makes the NPS proj model shallower. (DecoderSamplerV07)

V0717 adds validation and grad accumulation. Also clips the deterministic initial guess to (0,1). Makes the NPS proj model shallower. 
Removes Relu before latent space and input convs. Only best model is saved in Checkpoints (DecoderSamplerV17) (see run_set_2_convnext)

Run_set_4_convnext removes Relu bfore the first conv in UNet2. Also adds layernormalization before downsampling convolution.
Only best model is saved in Checkpoints

