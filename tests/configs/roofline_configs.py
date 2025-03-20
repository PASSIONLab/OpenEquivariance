from openequivariance.interface.tpp_creation_utils import SingleInstruction

roofline_configs = [
    SingleInstruction(L1, L2, L3, cm, f"[{i+1}]#{L1} x {L2} -> {L3} ({cm})")
    for i, (L1, L2, L3, cm) in enumerate([
        ("128x1e", "1x1e", "128x1e", "uvu"), 
        ("128x2e", "1x1e", "128x2e", "uvu"),
        ("128x3e", "1x3e", "128x3e", "uvu"),
        ("128x5e", "1x5e", "128x3e", "uvu"),
        ("128x5e", "1x3e", "128x5e", "uvu"),
        ("128x6e", "1x3e", "128x6e", "uvu"),
        ("128x7e", "1x4e", "128x7e", "uvu"),
        ("128x7e", "1x7e", "128x7e", "uvu"),
    ])
]