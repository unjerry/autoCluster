mkdir fig
python restartable.py regenerate
python restartable.py --train=deLinearSetupExp
python restartable.py --train=deLinearTrain
python restartable.py --train=deLinearGenerate
