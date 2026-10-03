"""Stage 0: run DM2's pretrained unconditional a-SiO2 model and compare g(r).  [Ticket B-1]

Needs a GPU and a separate environment matching DM2 (python 3.10, torch 2.5, e3nn==0.4.4,
`pip install -e DM2`). With torch >= 2.6 load the pickled model with weights_only=False.
Steps: clone https://github.com/digital-synthesis-lab/DM2, run
demo/demo_generating/denoise_generate_unconditional.py on the 300-atom demo (~2 min on an
A6000), then compare the Si-O / O-O / Si-Si g(r) of the output with the training
structures in demo/demo_training/simu_data/. Save the figure to results/stage0/.
"""

if __name__ == "__main__":
    raise NotImplementedError("Ticket B-1")
