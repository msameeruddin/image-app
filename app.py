import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq

import plotly.express as px

from dash.dependencies import (Input, Output, State)
from image_ops_scratch import (ImageOperations, read_image_string)
from image_morphs_scratch import MorphologicalTransformations

########################################
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css'
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions'] = True
app.title = 'Image Processing App'
server = app.server
########################################

image_ops = ['None', 'Equalize', 'Flip', 'Mirror', 'Binarize', 'Invert', 'Solarize']
image_morphs = ['None', 'Erode', 'Dilate', 'Open', 'Close', 'Gradient', 'Boundary Extraction']

tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '10px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '5px solid #d6d6d6',
    'borderBottom': '3px solid #d6d6d6',
    'backgroundColor': '#7E8483',
    'color': 'white',
    'padding': '6px'
}

app.layout = html.Div([
    html.Meta(charSet='UTF-8'),
    html.Meta(name='viewport', content='width=device-width, initial-scale=1.0'),

    html.Div([
        html.Div(
            id='title-app', 
            children=[
                html.H3(app.title)
            ],
            style={'textAlign' : 'center', 'paddingTop' : 30}
        ),
        html.Div([
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '70px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px',
                    'backgroundColor': '#F0F1F1'
                },
                multiple=True
            ),
        ], style={'paddingTop' : 50}),
        html.Div([
            dcc.Tabs(
                id='image-processors-tabs',
                value='operators',
                children=[
                    dcc.Tab(
                        label='Operations',
                        value='operators',
                        style=tab_style,
                        selected_style=tab_selected_style,
                        children=[
                            html.Div([
                                daq.ToggleSwitch(
                                    id='image-mode',
                                    size=60,
                                    label='Gray Scale',
                                    labelPosition='top',
                                    color='#717171',
                                    value=False,
                                )
                            ], style={'paddingTop' : 30, 'paddingBottom' : 10}),
                            html.Div([
                                dcc.RadioItems(
                                    id='in-operation', 
                                    options=[{'label' : op, 'value' : op.lower()} for op in image_ops],
                                    value='none'
                                ),
                            ], className='select-operation')
                        ]
                    ),
                    dcc.Tab(
                        label='Transformations',
                        value='transformers',
                        style=tab_style,
                        selected_style=tab_selected_style,
                        children=[
                            html.Div([
                                html.P('Morph level - '),
                                dcc.Input(id='morph-level', type='number', placeholder='Enter Morph Level - ', value=3),
                                html.Div([
                                    dcc.RadioItems(
                                        id='in-transformation', 
                                        options=[{'label' : tr, 'value' : tr.lower()} for tr in image_morphs],
                                        value='none'
                                    ),
                                ], style={'paddingTop' : 20})
                            ], className='select-operation')
                        ]
                    )
                ]
            )
        ], className='tab-div')
    ], className='flex-item-left'),

    html.Div(id='result-in-out-image', className='flex-item-right'),

], className='flex-container')


def parse_contents(contents, filename, date):
    image_mat = read_image_string(contents=contents)
    return image_mat


@app.callback(
    Output('result-in-out-image', 'children'), 
    [Input('image-processors-tabs', 'value')]
)
def set_output_layout(which_tab):
    if which_tab == 'operators':
        in_out_image_div = html.Div([
            html.Div( 
                children= [
                    html.H5('Image Used - Output'),
                    dcc.Loading(
                        id='loading-op',
                        type='dot',
                        children=html.Div(id='output-image-op')
                    )
                ],
                style={'textAlign' : 'center', 'paddingTop' : 50}
            )
        ])
    elif which_tab == 'transformers':
        in_out_image_div = html.Div([
            html.Div( 
                children= [
                    html.H5('Image Used - Output'),
                    dcc.Loading(
                        id='loading-morph',
                        type='dot',
                        children=html.Div(id='output-image-morph')
                    )
                ],
                style={'textAlign' : 'center', 'paddingTop' : 50}
            )
        ])
    return in_out_image_div


@app.callback(
    Output('output-image-op', 'children'), 
    [
        Input('upload-image', 'contents'), 
        Input('image-mode', 'value'), 
        Input('in-operation', 'value'), 
        # -------
        State('upload-image', 'filename'), 
        State('upload-image', 'last_modified'), 
    ]
)
def get_operated_image(contents, image_mode, operation, filenames, dates):
    if contents is not None:
        imsrc = parse_contents(contents, filenames, dates)
        imo = ImageOperations(image_file_src=imsrc)
        if (operation == 'equalize'):
            out_img = imo.equalize_this(gray_scale=True) if image_mode else imo.equalize_this()
        elif (operation == 'flip'):
            out_img = imo.flip_this(gray_scale=True) if image_mode else imo.flip_this()
        elif (operation == 'mirror'):
            out_img = imo.mirror_this(gray_scale=True) if image_mode else imo.mirror_this()
        elif (operation == 'binarize'):
            out_img = imo.binarize_this(gray_scale=True) if image_mode else imo.binarize_this()
        elif (operation == 'invert'):
            out_img = imo.invert_this(gray_scale=True) if image_mode else imo.invert_this()
        elif (operation == 'solarize'):
            out_img = imo.solarize_this(gray_scale=True) if image_mode else imo.solarize_this()
        else:
            out_img = imo.read_this(gray_scale=True) if image_mode else imo.read_this()

        out_image_fig = px.imshow(out_img, color_continuous_scale='gray') if image_mode else px.imshow(out_img)
        out_image_fig.update_layout(
            coloraxis_showscale=False, 
            width=600, height=400, 
            margin=dict(l=0, r=0, b=0, t=0)
        )
        out_image_fig.update_xaxes(showticklabels=False)
        out_image_fig.update_yaxes(showticklabels=False)

        output_result = html.Div([
            dcc.Graph(id='out-op-img', figure=out_image_fig)
        ], style={'paddingTop' : 50})

        return output_result


@app.callback(
    Output('output-image-morph', 'children'),
    [
        Input('upload-image', 'contents'), 
        Input('morph-level', 'value'), 
        Input('in-transformation', 'value'),
        # -------
        State('upload-image', 'filename'), 
        State('upload-image', 'last_modified'), 
    ]
)
def get_transformed_image(contents, level, transformation, filenames, dates):
    if contents is not None:
        imsrc = parse_contents(contents, filenames, dates)
        morph = MorphologicalTransformations(image_file_src=imsrc, level=level)
        level = 3 if level == None else level
        image_src = morph.read_this()

        if (transformation == 'erode'):
            out_img = morph.erode_image(image_src=image_src)
        elif (transformation == 'dilate'):
            out_img = morph.dilate_image(image_src=image_src)
        elif (transformation == 'open'):
            out_img = morph.open_image(image_src=image_src)
        elif (transformation == 'close'):
            out_img = morph.close_image(image_src=image_src)
        elif (transformation == 'gradient'):
            out_img = morph.morph_gradient(image_src=image_src)
        elif (transformation == 'boundary extraction'):
            out_img = morph.extract_boundary(image_src=image_src)
        else:
            out_img = image_src

        out_image_fig = px.imshow(out_img, color_continuous_scale='gray')
        out_image_fig.update_layout(
            coloraxis_showscale=False, 
            width=600, height=400, 
            margin=dict(l=0, r=0, b=0, t=0)
        )
        out_image_fig.update_xaxes(showticklabels=False)
        out_image_fig.update_yaxes(showticklabels=False)

        output_result = html.Div([
            dcc.Graph(id='out-morph-img', figure=out_image_fig)
        ], style={'paddingTop' : 50})

        return output_result


# if __name__ == '__main__':
#     app.run_server(debug=True)