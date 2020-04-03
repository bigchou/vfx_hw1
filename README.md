# VFX HW1

### Installation

~~~~
pip install -r requirements.txt
~~~~

### Execution

~~~~
# Reinhard 2005
python main.py --annofile exp/ex_2/task2_ss_ex2.txt --imgfolder exp/ex_2 --outfolder anyway --tone_method "Reinhard2005"

# Reinhard 2002 (Global Operator)
python main.py --annofile exp/ex_2/task2_ss_ex2.txt --imgfolder exp/ex_2 --outfolder anyway --tone_method "Reinhard2002Global"

# Reinhard 2002 (Local Operator)
python main.py --annofile exp/ex_2/task2_ss_ex2.txt --imgfolder exp/ex_2 --outfolder anyway --tone_method "Reinhard2002Local"

# support MTB
python main.py --annofile exp/ex_2/task2_ss_ex2.txt --imgfolder exp/ex_2 --outfolder anyway --tone_method "Reinhard2002Local" --MTB

# To learn more, please use the following commands
python main.py --help
~~~~


### Experiment

##### For Windows

~~~~
run.bat
~~~~



##### For Linux, OSX

~~~~
bash run.sh
~~~~



##### MTB Results

###### INPUT

<img src="result/MTB/origin.jpg" width="1014" height="169">

###### OUTPUT

<img src="result/MTB/MTB_result.jpg" width="1014" height="169">


##### HDR

| Red        | Green           | Blue  |
| ---------- |:---------------:| -----:|
| <img src="result/ex_2/ResponseCurve_ro.png" width="320" height="240">      | <img src="result/ex_2/ResponseCurve_go.png" width="320" height="240"> | <img src="result/ex_2/ResponseCurve_bo.png" width="320" height="240"> |
| <img src="result/ex_2/Red.png" width="320" height="240">      | <img src="result/ex_2/Green.png" width="320" height="240">      |   <img src="result/ex_2/Blue.png" width="320" height="240"> |

Note. To view the assembled high dynamic range image, plase drag [*.hdr](result/ex_2/HDR.exr) file to this [website](https://viewer.openhdr.org/).



##### Tone Reproduction

|        | Input         | Reinhard2002  | Reinhard2005  |
| ------ | ------------- |:-------------:| -------------:|
| Task 1 | <img src="exp/ex_3/DSCF5980.jpg" width="150" height="100">     | <img src="HDR_TASK1_2002/Reinhard2002Global.jpg" width="150" height="100"> | <img src="HDR_TASK1_2005/Reinhard2005.jpg" width="150" height="100"> |     
| Task 2 | <img src="exp/ex_1/DSCF6014.jpg" width="150" height="100">      | <img src="HDR_TASK2_2002/Reinhard2002Local.jpg" width="150" height="100"> | <img src="HDR_TASK2_2005/Reinhard2005.jpg" width="150" height="100"> |
| Task 3 | <img src="exp/ex_2/DSCF6030.jpg" width="150" height="100">      | <img src="HDR_TASK3_2002/Reinhard2002Local.jpg" width="150" height="100">      |   <img src="HDR_TASK3_2005/Reinhard2005.jpg" width="150" height="100"> |



### Reference

[1] G. Ward. Fast, robust image registration for compositing high dynamic range photographs from hand-held exposures.Journal of graphics tools, 8(2):17–30, 2003.

[2] P. DEBEVEC and J. MALIK. Recovering high dynamic range radiance maps from photographs.  In Computer graphics proceedings, annual conference series, pages 369–378. Association for Computing Machinery SIGGRAPH, 1997.

[3] E. Reinhard,  M. Stark,  P. Shirley,  and J. Ferwerda.   Photographic tone reproduction for digital images.  In Proceedings of the 29th annual conference on Computer graphics and interactive techniques, pages 267–276, 2002.

[4] E. Reinhard  and  K.  Devlin.   Dynamic  range  reduction  inspired by photoreceptor physiology. IEEE transactions on visualization and computer graphics, 11(1):13–24, 2005.

[5] R. Fattal, D. Lischinski, and M. Werman.   Gradient domain high  dynamic  range  compression. In Proceedings  of  the 29th annual conference on Computer graphics and interactive techniques, pages 249–256, 2002.
