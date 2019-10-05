# FEGAN
## progress
- downsample flow map to 1/2
- add self-attention after upsampling
- the discriminator should classify real distorted image as false(0)
- change dis layer to 6
- add smooth gradient constraint
- set idt to 0
- fix quiver and radial bug
- Refine flow map
  - add smooth angle constraint
  - add self flow constraint(the flow map should be consistent after rotating by 90, 180, 270 degree)
  - add cross flow constraint(F(x) = F(x'), x' is rotated from x by 90 degree)
  - add inter flow constraint(the flow should point to the center)
- fix test script
## TODO
1. Dis 試試no patch
2. Gen 試試直接in 256 out 128
3. Gen 試試 Edge

## dataset
- MCindoor_fisheye: https://drive.google.com/drive/folders/1tjl5HC2JHqI-eRlO4FKZzzB_bl0c-6X7?usp=sharing