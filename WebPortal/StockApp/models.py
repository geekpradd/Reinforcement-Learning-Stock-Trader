from django.db import models
from picklefield.fields import PickledObjectField

class User(models.Model) :
	key = models.CharField(max_length = 100)
	capital = models.IntegerField()
	initial_stocks = models.IntegerField()
	agent_type = models.IntegerField()
	progress = models.IntegerField()			# percent trading done
	graph_data = PickledObjectField(null=True)			# list of points for making the graphs

class File(models.Model):
	file = models.FileField()
	owner = models.ForeignKey(User, on_delete=models.CASCADE)
