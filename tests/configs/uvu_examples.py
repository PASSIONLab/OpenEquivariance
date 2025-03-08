from openequivariance.interface 
mace_conv = [
    ("128x0e+128x1o+128x2e", "1x0e+1x1o+1x2e+1x3o", "128x0e+128x1o+128x2e+128x3o", "mace-large"),
    ("128x0e+128x1o", "1x0e+1x1o+1x2e+1x3o", "128x0e+128x1o+128x2e", "mace-medium")
]

nequip_conv = [
    ('32x0o + 32x0e + 32x1o + 32x1e + 32x2o + 32x2e', '0e + 1o + 2e', '32x0o + 32x0e + 32x1o + 32x1e + 32x2o + 32x2e', 
            'nequip-lips'),
    ('64x0o + 64x0e + 64x1o + 64x1e', '0e + 1o', '64x0o + 64x0e + 64x1o + 64x1e',
            'nequip-revmd17-aspirin'),
    ('64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e', '0e + 1o + 2e', '64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e',
            'nequip-revmd17-toluene'),
    ('64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e + 64x3o + 64x3e',  '0e + 1o + 2e + 3o', '64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e + 64x3o + 64x3e', 
            'nequip-revmd17-benzene'),
    ('32x0o + 32x0e + 32x1o + 32x1e', '0e + 1o', '32x0o + 32x0e + 32x1o + 32x1e', 
            'nequip-water'),
]
