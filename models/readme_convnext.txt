multi-channel partial convs as before, no skip connections, layernormalization, biases turned True, Obs mask used for prior model, skip VAE

Runs with * in their name have smoothed land masked loss weights.

NPSproj made smaller in longitude.

V0417 introduces noise to decoder and trains with CRPS.

V0707 increases number of noise injections.  Also clips the deterministic initial guess to (0,1) for saved unet. Makes the NPS proj model shallower. (DecoderSamplerV07)

V0717 adds validation and grad accumulation. Also clips the deterministic initial guess to (0,1). Makes the NPS proj model shallower. 
Removes Relu before latent space and input convs. Only best model is saved in Checkpoints (DecoderSamplerV17) (see run_set_2_convnext)

V0911 has three noise injection levels, removes the extra singleconvnext layer unless skip-conv is on. Removes double and singleconvnext noise injection into the skip module. Edits Up blocks for adding noise. (run_set_3_convnext)

V1001 mainly injects noise in the mid_conv in Up blocks rather than ConvNext blocks and adds pad_ice in the last up block.(run_set_3_convnext)

V1031 is a change in code to incorporate cyclical beta annealing and the addition of ensemble spread as a conditioning field. Also, the lr scheduler is now cosine in place of linear! (could run wuth cvae_1001.py in run_set_4_convnext)

It also removes the cliping of the deterministic initial guess to (0,1) unless the UNet is pretrained and freeze deterministic is on. Finally, the low ress losses are used with stride 1. (runs with cvae_1031.py run_set_final_convnext)

## UNet2

Run_set_4_convnext in "UNet2" removes Relu bfore the first conv in UNet2. Also adds layernormalization before downsampling convolution.
Only best model is saved in Checkpoints

Run_set_5_convnext in "UNet2" can use ensmeble spread as input and uses cosine annealing for learning rate scheduler. 


Nore that clamped option is not added to the 1031 version in parallel modes.

