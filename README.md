# DS 340W - Course Project

This GitHub repository is for academic purposes only. The materials here are for a course project in Applied Data Science. DS 340W is an Applied Data Science class focused on research projects. All reference links will be provided to the original source. All code and data access credit goes to the original publishers.

## Contents

### Datasets

The **CS2CD (Counter-Strike 2 Cheat Detection)** dataset is an anonymized dataset comprised of Counter-Strike 2(CS2) gameplay at a variety of skill-levels with cheater annotations. 

Our project is based on this dataset, but with the use of supervised machine learning models (Logistic Regression & Random Forest instead of Transformer models. We hope to see if this change results in a positive or negative outcome.

Link to originally published data: (https://huggingface.co/datasets/CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection#loading-dataset) 

**Important Paths:**
- `data/no_cheater_present`: Folder containing data where no cheaters are present.
- `data/full_dataset/with_cheater_present`: Folder containing data with at least one cheater present.
- `README.md`: This documentation file
- `DS340W_ProjectImplementation/train_supervised.py`: Modified Implementation file
- `DS340W_ProjectImplementation/train_xgboost.py`: Modified Implementation file

### Code
`results/visualizations.ipynb` - This 'ipynb' file is the code that will run the full data analysis and produce multiple essential data visualizations.

-- Code was altered from original source. --
  
### Data source

The data is scraped from the website [csstats.gg](https://csstats.gg/) using the `ALL MATCHES` page as an entry point for scraping. This resulted in NUMBER `.dem` files. 

To extract the data from these files, the Python library demoparser2 was used[[github](https://github.com/LaihoE/demoparser)][[pypi](https://pypi.org/project/demoparser2/)]. 

Loading of the data as recommended in the section "[Loading dataset](#loading-dataset)" returns these types as well.

Link to the original parent paper: (https://ieeexplore.ieee.org/document/11114092)

### Citation

```bibtex
@misc{mille_mei_zhen_loo_2025,
  author       = { Mille Mei Zhen Loo and Gert Lužkov },
  title        = { CS2CD.Counter-Strike_2_Cheat_Detection (Revision 44e5129) },
  year         = 2025,
  url          = { https://huggingface.co/datasets/CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection },
  doi          = { 10.57967/hf/5654 },
  publisher    = { Hugging Face }
}
