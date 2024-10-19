mkdir fig
python restartable.py regenerate
python restartable.py --train=localPCAcircleSetup
python restartable.py --train=localPCAtrain
