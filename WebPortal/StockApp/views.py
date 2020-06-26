from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import User, File
import datetime, random, _thread, time, copy
import csv,io
import pandas as pd
from .ddqn import wrapper
from .DDPG.baselines import DDPGgive_results
def homepage(request) :
	if not request.session.session_key :
		request.session.save()

	return render(request, 'index.html')

def StartTrading(capital, initial_stocks, agent_type, files2, sesID) :
	files = copy.deepcopy(files2)
	data_set = []
	for file in files:
		df = pd.read_csv(file)
		data_set.append(df.sort_values('Date'))
		# print(list(df.columns))
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
	if agent_type=='0':
		print('x')
		ans[0],ans[1],ans[2] = DDPGgive_results(data_set,int(capital))
	elif agent_type =='1':
		ans[0],ans[1],ans[2] = DDPGgive_results(data_set,int(capital),initial_stocks)
	else:
		profitst, balancest, sharest, actionst, worthst, pricest = wrapper(data_set[0], int(capital), int(initial_stocks))
		ans[2] = [[]]
		for i in range(len(profitst)):
			ans[0].append([i+1, profitst[i]])
			ans[2][0].append([i, actionst[i]])

		for j in range(len(files)) :
			ans[1].append([])
			for i in range(len(profitst)):
				ans[1][j].append([i+1, sharest[i]])
	# print(ans[0])
	# Do some machaxxx RL and populate ans

	# Machaxxx RL :
	# for i in range(len(ans[0])) :
	# # 	ans[0].append([i+1, random.randrange(i, 2*i+1, 1)])
	# 	ans[2].append([i+1, 2*(random.random()-0.5)])

	# for j in range(len(files)) :
	# 	ans[1].append([])
	# 	for i in range(500) :
	# 		ans[1][j].append([i+1, random.randrange(0, 5000, 1)])

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

		# # Code for saving files in databse (not required as of now)
		# for file in files :
		# 	f = File(file = file, owner = u)
		# 	f.save()

		_thread.start_new_thread(StartTrading, (capital, initial_stocks, agent_type, files, sesID))

	time.sleep(0.5)

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
