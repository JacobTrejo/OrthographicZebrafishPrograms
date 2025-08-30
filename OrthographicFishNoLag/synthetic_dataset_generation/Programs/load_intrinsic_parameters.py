import yaml

def load_intrinsic_parameters(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    return (
        data['c_eye_u'],   # c_eye
        data['c_head_u'],  # c_head
        data['c_belly_u'], # c_belly
        data['d_eye_u'],   # d_eye
        data['eye_br_u'],  # eye_br
        data['head_br_u'], # head_br
        data['belly_br_u'],# belly_br
        data['eye_w_u'],   # eye_w
        data['eye_l_u'],   # eye_l
        0.3,   # eye_h
        data['head_w_u'],  # head_w
        data['head_l_u'],  # head_l
        0.53,  # head_h
        data['belly_w_u'], # belly_w
        data['belly_l_u'], # belly_l
        0.34, # belly_h
        data['seglen_u'],   # seglen
        data['ball_size_u'], # tail ball size
        data['ball_thickness_u'],
        data['tail_br_u']
    )
