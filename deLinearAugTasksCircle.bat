mkdir fig
python restartable.py regenerate
python restartable.py --train=deLinearAugSetupCircle
python restartable.py --train=deLinearAugTrain
python restartable.py --train=deLinearAugTrain
python restartable.py --train=deLinearAugTrain
@REM python restartable.py --train=deLinearGenerate
