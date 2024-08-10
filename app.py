import streamlit as st
import organism_v1 as sim


__author__ = '[Lukas Calmbach](lcalmbach@gmail.com)'
__version__ = '0.0.1'

# Initialize session state for settings if not already initialized
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'pop_size': 50,       # number of organisms
        'food_num': 100,      # number of food particles
        'gens': 50,           # number of generations
        'elitism': 0.20,      # elitism (selection bias)
        'mutate': 0.10,       # mutation rate
        'gen_time': 100,      # generation length (seconds)
        'dt': 0.04,           # simulation time step (dt)
        'dr_max': 720,        # max rotational speed (degrees per second)
        'v_max': 0.5,         # max velocity (units per second)
        'dv_max': 0.25,       # max acceleration (+/-) (units per second^2)
        'x_min': -2.0,        # arena western border
        'x_max': 2.0,         # arena eastern border
        'y_min': -2.0,        # arena southern border
        'y_max': 2.0,         # arena northern border
        'plot': True,         # plot final generation?
        'inodes': 1,          # number of input nodes
        'hnodes': 5,          # number of hidden nodes
        'onodes': 2           # number of output nodes
    }

# Sidebar with radio buttons for different pages
page = st.sidebar.radio("Navigation", ["Settings", "Calculate", "Analyse"])

# Impressum in the sidebar
def show_impressum():
    st.sidebar.markdown("### Impressum")
    st.sidebar.info(f"""
    Version: {__version__}\n
    [Streamlit](https://streamlit.io/)-app by: {__author__}  
    original code by: [Nathan Rooy](https://nathanrooy.github.io/posts/2017-11-30/evolving-simple-organisms-using-a-genetic-algorithm-and-deep-learning/)
    """)

# Title of the app
st.title("EvoSim")

# Settings Page
if page == "Settings":
    st.header("Simulation Settings")
    # Allow user to modify settings
    st.session_state.settings['pop_size'] = st.number_input('Population Size', min_value=1, max_value=500, value=st.session_state.settings['pop_size'])
    st.session_state.settings['food_num'] = st.number_input('Number of Food Particles', min_value=1, max_value=500, value=st.session_state.settings['food_num'])
    st.session_state.settings['gens'] = st.number_input('Number of Generations', min_value=1, max_value=500, value=st.session_state.settings['gens'])
    st.session_state.settings['elitism'] = st.slider('Elitism (Selection Bias)', min_value=0.0, max_value=1.0, value=st.session_state.settings['elitism'])
    st.session_state.settings['mutate'] = st.slider('Mutation Rate', min_value=0.0, max_value=1.0, value=st.session_state.settings['mutate'])

    st.header("Simulation Settings")
    st.session_state.settings['gen_time'] = st.number_input('Generation Length (seconds)', min_value=1, max_value=1000, value=st.session_state.settings['gen_time'])
    st.session_state.settings['dt'] = st.number_input('Simulation Time Step (dt)', min_value=0.001, max_value=1.0, value=st.session_state.settings['dt'], step=0.001)
    st.session_state.settings['dr_max'] = st.number_input('Max Rotational Speed (degrees/second)', min_value=1, max_value=3600, value=st.session_state.settings['dr_max'])
    st.session_state.settings['v_max'] = st.number_input('Max Velocity (units/second)', min_value=0.1, max_value=10.0, value=st.session_state.settings['v_max'])
    st.session_state.settings['dv_max'] = st.number_input('Max Acceleration (+/-) (units/second^2)', min_value=0.01, max_value=1.0, value=st.session_state.settings['dv_max'], step=0.01)

    st.header("Arena Settings")
    st.session_state.settings['x_min'] = st.number_input('Arena Western Border', value=st.session_state.settings['x_min'])
    st.session_state.settings['x_max'] = st.number_input('Arena Eastern Border', value=st.session_state.settings['x_max'])
    st.session_state.settings['y_min'] = st.number_input('Arena Southern Border', value=st.session_state.settings['y_min'])
    st.session_state.settings['y_max'] = st.number_input('Arena Northern Border', value=st.session_state.settings['y_max'])

    st.header("Organism Neural Net Settings")
    st.session_state.settings['inodes'] = st.number_input('Number of Input Nodes', min_value=1, max_value=100, value=st.session_state.settings['inodes'])
    st.session_state.settings['hnodes'] = st.number_input('Number of Hidden Nodes', min_value=1, max_value=100, value=st.session_state.settings['hnodes'])
    st.session_state.settings['onodes'] = st.number_input('Number of Output Nodes', min_value=1, max_value=100, value=st.session_state.settings['onodes'])

    st.session_state.settings['plot'] = st.checkbox('Plot Final Generation?', value=st.session_state.settings['plot'])

# Calculate Page (Placeholder)
elif page == "Calculate":
    if st.sidebar.button("Run Simulation"):
        with st.spinner("Running Simulation..."):
            sim.run(st.session_state.settings)

# Analyse Page (Placeholder)
elif page == "Analyse":
    st.header("Analysis Page")
    st.write("This is where the analysis would take place.")

show_impressum()