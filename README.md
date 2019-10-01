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
1. 把flow rot90deg 
2. MCindoor 加上sign 為了讓他學中心
3. 存real_A fake_B chess flow就好
## dataset
- MCindoor_fisheye: https://drive.google.com/drive/folders/1I9LpS8N5_z8ke_AT2zI6UVccm9vW21By?usp=sharing