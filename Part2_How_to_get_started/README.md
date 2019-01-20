# **Prerequisites for Part 2 Navigation Project** 

## Dependencies

Follow the instructions below to install all necessary dependencies.

### **Step 1:** Clone the [DRLND GitHub Repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) if you haven't already. 

Please follow the instructions in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in `README.md` at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

(_For Windows users_) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

### **Step 2:**  Download the Unity Environment

For this project, you will **not** need to install Unity - this is because Udacity has already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.

**(_For Windows users_)** Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

**(_For AWS_)** If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the Linux operating system above._)

### **Step 3**: Explore the Environment

After you have followed the instructions above, open `Navigation.ipynb` (located in the `p1_navigation/` folder in the DRLND GitHub repository) and follow the instructions to learn how to use the Python API to control the agent.


### **Jupyter Notebook**

If you are unfamiliar with Jupyter Notebooks, check out <A HREF="https://www.packtpub.com/books/content/basics-jupyter-notebook-and-python" target="_blank">Cyrille Rossant's Basics of Jupyter Notebook and Python</A> to get started.

Jupyter is an Ipython notebook where you can run blocks of code and see results interactively. All the code for this project is contained in a Jupyter notebook. Please create an IPython kernel for the drlnd environment first as discribed in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) installation instructions.

To start Jupyter in your browser, use terminal to navigate to your project directory and then run the following command at the terminal prompt (be sure you've activated your Python 3 **drlnd** environment as described in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) installation instructions!):

`> jupyter notebook`

A browser window will appear showing the contents of the current directory.  Click on the file called "xxx.ipynb".  Another browser window will appear displaying the notebook. Dont't forget to change the kernel to match the drlnd environment by using the drop-down `Kernel` menu before running code in a notebook.
