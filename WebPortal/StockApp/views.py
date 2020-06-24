from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import User, File
import datetime, random, _thread, time

def homepage(request) :
	if not request.session.session_key :
		request.session.save()

	return render(request, 'index.html')

def StartTrading(capital, initial_stocks, agent_type, files, sesID) :
	'''
	Updates the database by a list of three lists.
	First list contains tuples of (x, y) values corresponding to profit values.
	Second list contains n lists where n is the number of files uploaded by the user. Each of those n lists contains tuples of (x, y) values corresponding to the number of shares bought bought by agent of that company.
	Third list contains tuples of (x, y) values corresponding to the action taken by the agent.

	agent_type :
	0 : DDPG Multi, 1 : DDPG Single, 2 : DDQN Single, 3 : DRQN Single

	files is a list of one or more files depending on agent_type
	''' 

	ans = [[], [], []]

	# Do some machaxxx RL and populate ans

	# Machaxxx RL :
	for i in range(200) :
		ans[0].append([i+1, random.randrange(10, 10000, 1)])
		ans[2].append([i+1, 2*(random.random()-0.5)])

	for j in range(len(files)) :
		ans[1].append([])
		for i in range(200) :
			ans[1][j].append([i+1, random.randrange(0, 5000, 1)])

	usr = User.objects.get(key = sesID)
	usr.graph_data = ans
	usr.progress = 100
	usr.save()


def fileUpload(request) :
	if not request.session.session_key :
		request.session.save()

	files = request.FILES.getlist('files')
	capital = request.POST.get('capital')
	agent_type = request.POST.get('agentType')
	# 0 : DDPG Multi, 1 : DDPG Single, 2 : DDQN Single, 3 : DRQN Single

	initial_stocks = request.POST.get('stocks')
	try :
		initial_stocks = int(initial_stocks)
	except:
		initial_stocks = 0

	sesID = request.session.session_key
	is_valid = False

	## Write some code to check if the data is valid
	is_valid = True

	if is_valid :
		usr = User.objects.filter(key = sesID)
		if usr.exists() :
			u = usr[0]
			u.capital = capital
			u.initial_stocks = initial_stocks
			u.agent_type = agent_type
			u.progress = 0
			u.save()
		else :
			u = User(key=sesID, capital=capital, initial_stocks = initial_stocks, agent_type= agent_type, progress=0)
			u.save()

		# # Code for saving files in databse (not reqiquired as of now)
		# for file in files :
		# 	f = File(file = file, owner = u)
		# 	f.save()

		_thread.start_new_thread(StartTrading, (capital, initial_stocks, agent_type, files, sesID))

	data = {'valid' : is_valid}
	return JsonResponse(data)

def checkProgress(request) :
	sesID = request.session.session_key
	usr = User.objects.filter(key = sesID)

	if usr.exists() :
		u = usr[0]
		data = {'valid' : True, 'progress' : u.progress}
		if u.progress == 100 :
			data['graph_data'] = u.graph_data

		return JsonResponse(data)

	return JsonResponse({'valid' : False})
