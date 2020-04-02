@echo off
ECHO "========== MTB =========="
python MTB.py

ECHO "========== TASK1 =========="
SET TASK1_IMGFOLDER="exp/ex_3"
SET TASK1_ANNOTATION="exp/ex_3/task1_ss_ex2.txt"
SET TASK1_OUTFOLDER="HDR_TASK1"
python main.py --tone_method "Reinhard2002Global" --annofile %TASK1_ANNOTATION% --imgfolder %TASK1_IMGFOLDER%  --MTB --key 0.9 --outfolder %TASK1_OUTFOLDER%_2002 --drawHDR --drawCurve
python main.py --tone_method "Reinhard2005" --annofile %TASK1_ANNOTATION% --imgfolder %TASK1_IMGFOLDER%  --MTB --key -4.0 --contrast 0.5 --adaptation 1.0 --chromatic 0.5 --outfolder %TASK1_OUTFOLDER%_2005 --drawHDR --drawCurve

ECHO "========== TASK2 =========="
SET TASK2_IMGFOLDER="exp/ex_1"
SET TASK2_ANNOTATION="exp/ex_1/task2_ss_ex1.txt"
SET TASK2_OUTFOLDER="HDR_TASK2"
python main.py --tone_method "Reinhard2002Local" --annofile %TASK2_ANNOTATION% --imgfolder %TASK2_IMGFOLDER%  --MTB --key 0.9 --phi 5.0 --num_scale 1 --outfolder %TASK2_OUTFOLDER%_2002 --drawHDR --drawCurve
python main.py --tone_method "Reinhard2005" --annofile %TASK2_ANNOTATION% --imgfolder %TASK2_IMGFOLDER%  --MTB --key -4.0 --contrast 0.5 --adaptation 1.0 --chromatic 0.5 --outfolder %TASK2_OUTFOLDER%_2005 --drawHDR --drawCurve

ECHO "========== TASK3 =========="
SET TASK3_IMGFOLDER="exp/ex_2"
SET TASK3_ANNOTATION="exp/ex_2/task2_ss_ex2.txt"
SET TASK3_OUTFOLDER="HDR_TASK3"
python main.py --tone_method "Reinhard2002Local" --annofile %TASK3_ANNOTATION% --imgfolder %TASK3_IMGFOLDER%  --MTB --key 0.9 --phi 5.0 --num_scale 1 --outfolder %TASK3_OUTFOLDER%_2002 --drawHDR --drawCurve
python main.py --tone_method "Reinhard2005" --annofile %TASK3_ANNOTATION% --imgfolder %TASK3_IMGFOLDER%  --MTB --key -4.0 --contrast 0.5 --adaptation 1.0 --chromatic 0.5 --outfolder %TASK3_OUTFOLDER%_2005 --drawHDR --drawCurve
PAUSE