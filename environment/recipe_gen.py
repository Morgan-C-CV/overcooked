import random
def generate_recipe(menu):
    '''Return random recipe'''
    if len(menu) >= 5:
        return None
    else:
        return random.randint(0, 7)

if __name__ == '__main__':
    for i in range(30):
        print(generate_recipe([0, 1, 2]))
