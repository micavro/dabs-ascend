"""Only imports retained source; no model/API operation is performed."""
import os
from pathlib import Path
from wfsched.legacy import load_original, load_profile
ROOT=Path(os.environ.get('WF_SOURCE_ROOT','/home/wmy/sci/spra_paper_experiments_github'))
ORIGINAL=load_original(ROOT)
_,MODEL,PRIORS,MANIFEST=load_profile(ORIGINAL,ROOT)
SOURCE=ROOT/'spra_stage19_musique_joint_v1'
CFG=dict(MANIFEST['profile_configuration']);CFG['base_cap']=4096
