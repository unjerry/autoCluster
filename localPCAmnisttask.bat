mkdir fig
python restartable.py regenerate
python restartable.py --train=localPCAmnistSetup
python restartable.py --train=localPCAtrain
