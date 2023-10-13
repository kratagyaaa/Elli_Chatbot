


update requirement file :pip freeze > requirements.txt

install package from requirement file :
pip install -r requirements.txt




1.   go to your project Folder 
2.  if venv not exist use python -m venv venv   otherrwise skip this step
2.  got to project folder then activate => venv venv\scripts\activate
4.  install packages :
    python -m pip install --upgrade pip
    pip install -r requirements.txt

Flask server setup and run :https://code.visualstudio.com/docs/python/tutorial-flask

current test URLS :
http://127.0.0.1:5000/test     GET
http://127.0.0.1:5000/setup    GET
http://127.0.0.1:5000/chatbot  POST


To run server :
or use : flask --app MainApp.py  run




    <!-- pip install python-dotenv
	pip install -q langchain==0.0.150 pypdf pandas matplotlib tiktoken textract transformers openai faiss-cpu tensorflow waitress flask -->
	
	