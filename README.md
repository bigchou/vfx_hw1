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