import os
from sacred import Ingredient

project = Ingredient('project')


@project.config
def default_config():
    name = 'dcase24_task8'


@project.capture
def get_project_name(name):
    return name