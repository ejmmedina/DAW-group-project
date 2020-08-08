#!/usr/bin/env python
# coding: utf-8

# In[8]:


from IPython.display import display, IFrame, HTML
import os

def show_app(app, port=9999, width=900, height=700):
    host = 'localhost'
    url = f'http://{host}:{port}'

#     display(HTML(f"<a href='{url}' target='_blank'>Open in new tab</a>"))
#     display(IFrame(url, width=width, height=height))
    app.css.config.serve_locally = True
    app.scripts.config.serve_locally = True
    return app.run_server(debug=False, host=host, port=port)


# In[2]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import plotly.graph_objs as go
import pandas as pd
import re
import base64
from PIL import Image
from io import BytesIO
import numpy as np
from dash.dependencies import Input, Output, State


# In[3]:


# assets_folder directory where you put css and js
external_stylesheets = ["https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css", "https://codepen.io/chriddyp/pen/brPBPO.css"]
app = dash.Dash('__name__', assets_folder='static/', external_stylesheets=external_stylesheets)


# In[4]:


from tensorflow.keras.models import load_model


# In[12]:



app.callback_map = {} # erases all callbacks
app.layout = html.Div([
    html.Div([
        html.Br(),
        html.Br(),
        html.Div([
            html.H2('Over 10 million Filipinos', id='fils1'),
            html.H2('are depending their livelihood in rice farming. ', id='fils2'),
            html.H2('And these farmers experienced', id='farmers'),
            html.H1('an estimated average of 37% yield loss', id='loss'),
            html.H2('due to pests and diseases.', id='pests'),

        ]),
        html.Br(),
        html.Br(),
    ], id='container0'),
    html.Br(),
    html.H1('How can we help our rice farmers maximize their yield production?', id='p1'),
    html.Br(),
    html.Div([
        html.Br(),
        html.Br(),
        html.H2('Rice Leaf Disease Identifier', id='header1'),
        html.Br(),
        html.Br(),
        html.P('''Just upload an image of a rice leaf below and get to know the disease
               it posses and the remedies to prevent further loss of rice''', id='p2'),
        html.Br(),
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select a File')
            ])),
        html.Br(),
        html.H2('Important note:', id='note'),
        html.P('''Try uploading images with single leaf at a time as the
                algorithm won't predict images with multiple leaves.''',
               id='p3'),
        html.Br(),
    ], id='container1'),
    html.Br(),
    html.Div([
        html.Div(id='output-image-upload')
    ], id="upload-image-container"),
    html.Br(),
    html.Div(
        html.Button("Predict"),
        id='predict'
    ),

    html.Br(),
    html.Div(id='diagnosis')
])

@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename')])
def update_output(contents, filename):
    return [html.Div([
        html.H5(filename),
        html.Img(src=contents, height="256px"),
    ])]
@app.callback(Output('diagnosis', 'children'),
             [Input('predict', 'n_clicks')],
             [State('upload-image', 'contents')])
def diagnose_disease(btn, contents):
    if contents:
        model = load_model("data/models/best_model_5conv2dense.h5") #nag-eerror pag sa labas ng function ko dinefine
        image_data = re.sub('^data:image/.+;base64,', '', contents)
        im = Image.open(BytesIO(base64.b64decode(image_data)))
        pred = model.predict(np.array(im.resize((256, 256)))[None,:,:,:]/225)[0]
        classes = ["Brown spot", "Healthy", "Hispa", "Leaf blast"]
        text_out = [[html.B("Brown spot"), ''' is is a fungal disease commonly occurred in unflooded
                    and nutrient-deficient soil, or in soils that accumulate toxic substances.
                    Its most distinguishable feature is the big spots on the leaves which could
                    possibly damage the whole leaf.

                    Here are some of the solutions in preventing brown spot from further damaging the plants:''',
                     html.Ul([html.Li("Monitor soil nutrients and apply required fertilizers and nutrients."),
                              html.Li("Use resistant varieties of rice, use fungicides, or treat seeds with hot water (53-54degC) for 10-12 minutes before planting."),
                              html.Li("To increase effectiveness of treatment, pre-soak seeds in cold water for eight hours.")])],
                   ['''Congratulations! Your leaf is ''', html.B("Healthy")],
                   [html.B("Hispa"), ''' is a pest disease which commonly occurs when plant is
                   in or near grassy weeds or due to heavily fertilized fields.
                   This is dangerous to rice plants as it scrapes the upper surface
                   of the leaf blades leaving only the lower epidermis.
                   Here are some of the solutions in preventing the rice hispa from further damaging the plants:''',
                    html.Ul([html.Li("Avoid over fertilizing the field"),
                              html.Li("Plant rice with more space in-between"),
                              html.Li("To prevent egg laying of the pests, cut the shoot tips. Clipping and burying shoots in the mud can reduce grub populations by 75−92%")])],
                   [html.B("Leaf blast"), ''' is a fungal disease occurs in low soil moisture,
                   frequent and prolonged periods of rain shower, and cool temperature in the daytime.
                   It can affect all parts of rice plant and it easily occurs whenever there is presence
                   of spores in any part.
                    Here are some of the solutions in preventing the leaf blast from further damaging the plants:''',
                    html.Ul([html.Li("Plant resistant varieties of rice. "),
                              html.Li("Adjust planting time. Sow seeds early, when possible, after the onset of the rainy season."),
                              html.Li("Split nitrogen fertilizer application in two or more treatments. Excessive use of fertilizer can increase blast intensity."),
                              html.Li("Flood the field as often as possible. Silicon fertilizers (e.g., calcium silicate) can be applied to soils that are silicon deficient to reduce blast. Systemic fungicides like triazoles and strobilurins can be used judiciously to control blast")])]
                    ]
        text_classes = dict(zip(classes,text_out))
        out_elem = [html.P(text_out[pred.argmax()]), ]
        threshold = 0.1
        if pred.argmax()==1 and (pred>threshold).sum()>1:
            out_elem.append("However, there is still a small chance of the following diseases:")
            possible_disease = [i for i in np.array(classes)[pred>threshold] if i!="Healthy"]
            
            for i in possible_disease:
                text_to_out = text_classes[i]
                
                text_to_out.insert(1, html.B(f" ({100*pred[classes.index(i)]:.2f}%)"))
                out_elem.append(html.P(text_to_out))
#         probs = [[html.P(f"{100*prob:.2f}% {disease}"), html.Br()] for prob, disease in zip(pred, classes)]
#         elems = html.Div([i for j in probs for i in j])
#         return [*out_elem, elems]
        return out_elem

# # Dash CSS
# app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
# # Loading screen CSS
# app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})
show_app(app)


# In[ ]:




