mkdir fig
python restartable.py regenerate
python restartable.py --train=localPCASetup
python restartable.py --train=localPCAtrain
