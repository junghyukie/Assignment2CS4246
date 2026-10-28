# CSXX46 Assignment 2: Colab student package

This folder contains the two student notebooks and every local code/data file they use. The notebooks install `pyRDDLGym==2.7` and `rddlrepository==2.2` from PyPI when you run their setup cells; an internet connection is required for that first installation.

## Open in Google Colab

1. Extract `CSXX46-A2-Colab-Student.zip` on your computer. Upload the **entire** `CSXX46A2-Student` folder to the top level of **My Drive**, keeping all notebooks, helper files, and `selfDefinedEnvs/` together.
2. In Google Drive, open each template notebook **from inside** `CSXX46A2-Student`. Do not upload a notebook by itself into a separate Colab session.
3. In Colab, select **Runtime > Change runtime type > GPU** (for example, T4 GPU), then run the notebook's setup cells in order. The code looks for this folder automatically, so you do not need to edit RDDL paths.

If you use local Jupyter instead, extract the ZIP and start Jupyter from inside `CSXX46A2-Student`.

## Files

- `ElevatorEnv.ipynb`: environment and action-space guide.
- `Model-FreeRL-Template.ipynb`: Problem 1, model-free PPO.
- `Model-BasedRL-Template.ipynb`: Problem 2, model-based PPO.
- `utils.py`, `MyWrappers.py`, and `selfDefinedEnvs/`: local helpers and RDDL files.
- `requirements-colab.txt`: Python dependencies for reference or manual installation.

The two template notebooks intentionally contain task placeholders. Complete Problem 1 before adapting its PPO helper functions in Problem 2. Training saves `model_free_agent.pth`, `model_based_agent.pth`, `transition_model.pth`, and `reward_model.pth` in this folder. No pretrained checkpoint is included; submit checkpoints trained by your own completed notebooks, not the old repository checkpoints.

During MB-PPO training, each live refresh shows a diagnostic dashboard. Its **Total Reward** panel gives real-environment episode returns; the other panels show PPO losses, learned-model losses, clipping/KL, and actor/critic gradient norms. The notebook also shows a compact final summary and evaluation returns. Use these trends as sanity checks; grading evaluates the saved agents separately on fixed-seed real-environment episodes.

Submit one `groupXX_assignment2.zip` per team, replacing `XX` with your Canvas group number. Place the two completed template notebooks **with executed graph outputs** and the four named `.pth` checkpoints directly at the ZIP root. Follow the course platform for the deadline and homework submission.
