# Aldone

Aldone as in "AI-Done" is a tool for personal reminders and grocery list management.
This is a proof of concept of a product that could save you time every day by allowing you
to smoothly interact with it on the fly.  

## A short demo
https://github.com/faustozamparelli/Aldone/assets/105665123/fca0187f-2348-4fcf-a4c7-27c762c2331b  
  
Aldone is a react web app that interacts with a node.js and a python server doing most of the magic.
After a log-in you will be able to speak to it to add tasks to your grocery list, to-do list,
or make questions about reminders you saved earlier.
Furthermore, you could ask to split a task in sub-tasks or approximate how long a task will take.
Aldone is able to detect fully by himself weather you are asking him to add something to your grocery list or to-do list.

The coolest thing is that once you will come home from the supermarket with tired legs and arms from carrying bags, you won't have to manually remove everything from this digital grocery list but you can use your phone camera to detect what foods you actually bought and then Aldone will remove them for you. Other use cases for this include checking what's already in your kitchen or marking products as bought after delegating your groceries to someone else.

## This is a visual representation of all the agents that made up Aldone <img width="1230" alt="diagram" src="https://github.com/faustozamparelli/Aldone/assets/105665123/1219864e-c94d-49d5-b138-0cb50d636a06">

Before running the code, create the `.env` file and set the `OPENAI_API_KEY` variable with your OpenAI key.

To run the development server:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with **Google Chrome** to see the result.

To run the python server:

```bash
./setup_py_venv
source .venv/bin/activate
python src/py/server.py
```

Open [http://localhost:3001](http://localhost:3001) with your browser to see the result.

---

### To learn more please feel free to read the report here:

### https://github.com/faustozamparelli/Aldone/blob/main/report/Aldone.pdf

### or the presentation here:

### https://github.com/faustozamparelli/Aldone/blob/main/presentation/Aldone.pdf
