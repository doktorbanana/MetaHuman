# MetaHuman
Posthumanity makes music.

Project by Fynn Freist and Peter Martin.

For MetaHuman two neural networks have been trained on YouTube songs and learned how to sing. A closed loop was created, where the two networks sing to each other and learn from what they hear. If this process is repeated often enough, at some point there will be more machine-songs than human-songs in the training-set of the networks - at this point they've learned more from each other, then from humans. We hope, that this leads to "meta-" or "post-human" music. 

This project is part of the course "Musikinformatik" held in 2022 by Dennis Scheiba, Marcus Schmickler and Julian Rohrhuber at the institute of music and media in Düsseldorf. The Code is heavily based on the code and tutorials of Dennis Scheiba and Valerio Velardo. They can be found here:

https://capital-g.github.io/musikinformatik-sose2021/

https://github.com/capital-G/musikinformatik-sose2021

https://www.youtube.com/c/ValerioVelardoTheSoundofAI


## How to:

If you only want to get a quick overview of the project and don't care about the technical implementation, please download the folder "example files" and listen to the output of our two networks. In the two audio files you can hear one of the networks trying to sing a song it was trained on - Hey Jude by The Beatles and Ave Maria. The Video files show the networks singing to each other. You can find a videos of the two networks talking to each other here: https://drive.google.com/drive/folders/183Xb4NSxg_An0ehmYINbBRmoD4Kh4UoY?usp=sharing

If you are interested in the idea of the project, you can read the "Konzept.pdf" (only in german).

If you are interested in the technical implementation, please follow these steps:

1. Clone this repository. You need pip and virtualenv installed. 

2. Navigate to the project directory and create a new virtual enviornment with the following command: 
  ```bash
  virtualenv venv
  ```
  Now start your new virtual enviornment:
  
  ```bash
  # on linux/macOS
  source venv/bin/activate
  # on windows
  .\venv\Scripts\activate
  ```

3. Install the required modules with this command:
  ```bash
  pip3 install -r requirements.txt
  ```
  
4. Now open Jupyter Notebook with this command:
  ```bash
  jupyter lab
  ```

  (If you are not familiar with all of this, you can find a more detailed and very good explanation here: https://capital-g.github.io/musikinformatik-sose2021/docs/course-info/setup.html. Just replace https://github.com/capital-G/musikinformatik-sose2021.git in the first step with https://github.com/doktorbanana/MetaHuman.git You can skip the SuperCollider installation)

  You should now see seven different Notebooks in your browser (Notebooks 0-6). You can go through the code and our documentation. Start with Notebook "0 Preface". If you are only interested in the result you can skip notebooks 1-5 and jump to Notebook 6. 

5. To run the code in Notebook 6, you need our trained models. You can get them here: https://drive.google.com/drive/folders/1IU1-62HpnQDz7yyQlBsNWtEvn7Ym5lgc?usp=sharing
Please download all folders in the "data_and_models" folder and paste them in the "data_and_models" folder in the project directory (create it if it doesnt exist). 

6. To run the code in Notebooks 1-5, you need training data. If you want to train your own model, you can use any collection of wav-files as training data. Note that the process works better for single instrument stems (as described in Notebook "1 Data Crawling". You will need to change the paths in the notebooks to match your location. We've commented on that in the Notebook.



