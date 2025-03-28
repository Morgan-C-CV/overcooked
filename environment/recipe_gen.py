import random
def generate_recipe(menu):
    '''Return random recipe'''
    if len(menu) >= 5:
        return None
    else:
        return random.randint(0, 7)
