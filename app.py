# Import libraries
import joblib
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

# Load your trained model
model_pipeline = joblib.load('car_price_model_pipeline.pkl')

# Initialize Dash app
app = dash.Dash(__name__, assets_folder='assets')
server = app.server 

# Load data for analytics
car_price_df = pd.read_csv("used_car_price_dataset_extended.csv")

# Define custom CSS styles
app.layout = html.Div([
    # Navigation Bar
    html.Nav([
        html.Div([
            # Logo
            html.Div([
                html.Img(src='/assets/image4.png', style={'height': '40px', 'marginRight': '10px'}),
                html.H3("AfriCarModel", style={'color': '#FFFFFF', 'margin': '0', 'fontSize': '1.5rem', 'fontWeight': '700'})
            ], style={'display': 'flex', 'alignItems': 'center'}),
            
            # Navigation Links
            html.Div([
                html.A("Home", href="#home", style={'color': '#FFFFFF', 'textDecoration': 'none', 'margin': '0 20px', 'fontWeight': '500', 'transition': 'color 0.3s'}),
                html.A("Price Estimate", href="#prediction-section", style={'color': '#FFFFFF', 'textDecoration': 'none', 'margin': '0 20px', 'fontWeight': '500'}),
                html.A("Analytics", href="#analytics-section", style={'color': '#FFFFFF', 'textDecoration': 'none', 'margin': '0 20px', 'fontWeight': '500'}),
                html.A("About", href="#about-section", style={'color': '#FFFFFF', 'textDecoration': 'none', 'margin': '0 20px', 'fontWeight': '500'}),
                html.A("Contact", href="#contact-section", style={'color': '#FFFFFF', 'textDecoration': 'none', 'margin': '0 20px', 'fontWeight': '500'})
            ], style={'display': 'flex', 'alignItems': 'center'})
        ], style={
            'display': 'flex', 
            'justifyContent': 'space-between', 
            'alignItems': 'center',
            'maxWidth': '1200px',
            'margin': '0 auto',
            'padding': '0 20px'
        })
    ], style={
        'backgroundColor': 'rgba(26, 46, 69, 0.95)',  # Muted Navy
        'padding': '15px 0',
        'position': 'fixed',
        'top': '0',
        'width': '100%',
        'zIndex': '1000',
        'backdropFilter': 'blur(10px)',
        'boxShadow': '0 2px 20px rgba(0,0,0,0.1)'
    }),
    
    # Hero Header Section
    html.Div([
        html.Div([
            html.H1("AfriCarModel", 
                    style={
                        'color': '#FFFFFF', 
                        'margin': '0 0 10px 0', 
                        'marginTop': '-150px',
                        'fontSize': '4rem', 
                        'fontWeight': '800', 
                        'letterSpacing': '-2px', 
                        'textShadow': '2px 2px 4px rgba(0,0,0,0.3)',
                        'zIndex': '2',
                        'position': 'relative'
                    }),
            html.H2("Instantly know your car’s true value with an accurate market price powered by advanced AI trained on real vehicle data", 
                    style={
                        'color': '#FFFFFF', 
                        'margin': '0 0 15px 0', 
                        'fontSize': '2.0rem', 
                        'fontWeight': '600', 
                        'opacity': '0.95',
                        'zIndex': '2',
                        'position': 'relative'
                    }),
            html.P("Enter your car details and instantly get an estimated market value of your car", 
                   style={
                       'color': '#EDEDED',  # Light Gray
                       'margin': '0 0 30px 0', 
                       'fontSize': '1.1rem', 
                       'lineHeight': '1.6', 
                       'maxWidth': '700px', 
                       'margin': '0 auto 30px auto',
                       'zIndex': '2',
                       'position': 'relative'
                   }),
            html.A(
                html.Button("Get My Car Price", 
                            style={
                                'backgroundColor': '#007BFF',  # Vibrant Blue
                                'color': '#FFFFFF',
                                'border': 'none',
                                'padding': '18px 35px',
                                'fontSize': '1.3rem',
                                'fontWeight': '600',
                                'borderRadius': '50px',
                                'cursor': 'pointer',
                                'transition': 'all 0.3s ease',
                                'boxShadow': '0 6px 25px rgba(0, 123, 255, 0.4)',
                                'textTransform': 'uppercase',
                                'letterSpacing': '1px',
                                'transform': 'translateY(0)',
                                'animation': 'pulse 2s infinite',
                                'zIndex': '2',
                                'position': 'relative'
                            }),
                href="#prediction-section"
            )
        ], style={
            'textAlign': 'center', 
            'maxWidth': '900px', 
            'margin': '0 auto', 
            'zIndex': '2', 
            'position': 'relative',
            'padding': '0 20px'
        })
    ], id='home', style={
        'backgroundImage': 'url(/assets/image2.png)',  
        'backgroundSize': 'cover',                 
        'backgroundPosition': 'center',            
        'backgroundRepeat': 'no-repeat',           
        'padding': '120px 20px 80px 20px',
        'marginTop': '70px',
        'minHeight': '600px',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center',
        'position': 'relative',
        'overflow': 'hidden',
        'zIndex': '1',
        'boxShadow': 'inset 0 0 0 1000px rgba(0,0,0,0.4)'  
    }),
    
    # About Section
    html.Div([
        html.Div([
            html.H2("What is AfriCarModel?", 
                    style={
                        'color': '#1E1E1E',  # Dark Charcoal
                        'textAlign': 'right',  
                        'marginBottom': '40px', 
                        'fontSize': '2.8rem', 
                        'fontWeight': '700',
                        'marginTop': '-30px',
                    }),
            html.Div([
                html.Div([
                    html.Img(src='/assets/image3.png', style={
                        'width': '100%',
                        'height': '100%',  
                        'objectFit': 'cover',
                        'borderRadius': '15px',
                        'marginBottom': '0px'
                    })
                ], style={
                    'width': '45%', 
                    'display': 'inline-block', 
                    'verticalAlign': 'top',
                    'height': '100%',
                    'marginTop': '-40px'
                }),
                html.Div([
                    html.P("AfriCarModel is an intelligent pricing tool that revolutionizes how car buyers, sellers, and dealers determine the true market value of any used vehicle in Africa.", 
                           style={'color': '#1E1E1E', 'fontSize': '1.3rem', 'lineHeight': '1.8', 'marginBottom': '25px', 'textAlign': 'justify', 'marginTop': '-20px'}),
                    html.P("Using advanced machine learning algorithms trained on thousands of vehicle listings, AfriCarModel predicts fair prices based on comprehensive real-world data analysis.", 
                           style={'color': '#1E1E1E', 'fontSize': '1.3rem', 'lineHeight': '1.8', 'marginBottom': '25px', 'textAlign': 'justify', 'marginTop': '-20px'}),
                    html.P("Whether you're selling a car, buying one, or just curious about your vehicle's value, AfriCarModel provides insights you can trust with confidence.", 
                           style={'color': '#1E1E1E', 'fontSize': '1.3rem', 'lineHeight': '1.8', 'textAlign': 'justify', 'marginTop': '-20px'})
                ], style={
                    'width': '50%', 
                    'display': 'inline-block', 
                    'verticalAlign': 'top', 
                    'paddingLeft': '40px',
                    'height': '100%'
                })
            ], style={
                'display': 'flex',
                'alignItems': 'stretch'
            })
        ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '0 20px'})
    ], id='about-section', style={
        'backgroundColor': '#EDEDED',  # Light Gray
        'padding': '80px 20px',
        'marginBottom': '0px'
    }),
    
    # Main Content Container
    html.Div([
        # Prediction Section
        html.Div([
            html.H2("Enter Your Car Details to Get an Instant Price Estimate", 
                   style={'color': '#1E1E1E', 'textAlign': 'center', 'marginBottom': '50px', 'fontSize': '2.5rem', 'fontWeight': '600'}),
            # Input Form Section
            html.Div([
                # Form Grid
                html.Div([
                    # Left Column
                    html.Div([
                        # Make Year
                        html.Div([
                            html.Label("Make Year", style={'fontWeight': '600', 'color': '#1E1E1E', 'marginBottom': '8px', 'display': 'block'}),
                            dcc.Input(
                                id='input1',
                                type='number',
                                placeholder='e.g., 2018',
                                value=2018,
                                min=1995,
                                max=2023,
                                style={
                                    'width': '100%',
                                    'padding': '15px 18px',
                                    'border': '2px solid #EDEDED',
                                    'borderRadius': '10px',
                                    'fontSize': '16px',
                                    'transition': 'all 0.3s ease',
                                    'backgroundColor': '#FFFFFF'
                                }
                            )
                        ], style={'marginBottom': '25px'}),
                        
                        # Mileage
                        html.Div([
                            html.Label("Mileage (kmpl)", style={'fontWeight': '600', 'color': '#1E1E1E', 'marginBottom': '8px', 'display': 'block'}),
                            dcc.Input(
                                id='input2',
                                type='number',
                                placeholder='e.g., 15.5',
                                value=15,
                                min=5,
                                max=35,
                                step=0.1,
                                style={
                                    'width': '100%',
                                    'padding': '15px 18px',
                                    'border': '2px solid #EDEDED',
                                    'borderRadius': '10px',
                                    'fontSize': '16px',
                                    'backgroundColor': '#FFFFFF'
                                }
                            )
                        ], style={'marginBottom': '25px'}),
                        
                        # Engine CC
                        html.Div([
                            html.Label("Engine CC", style={'fontWeight': '600', 'color': '#1E1E1E', 'marginBottom': '8px', 'display': 'block'}),
                            dcc.Input(
                                id='input3',
                                type='number',
                                placeholder='e.g., 1500',
                                value=1500,
                                min=800,
                                max=5000,
                                style={
                                    'width': '100%',
                                    'padding': '15px 18px',
                                    'border': '2px solid #EDEDED',
                                    'borderRadius': '10px',
                                    'fontSize': '16px',
                                    'backgroundColor': '#FFFFFF'
                                }
                            )
                        ], style={'marginBottom': '25px'}),

                        # Fuel Type
                        html.Div([
                            html.Label("Fuel Type", style={'fontWeight': '600', 'color': '#1E1E1E', 'marginBottom': '8px', 'display': 'block'}),
                            dcc.Dropdown(
                                id='input4',
                                options=[
                                    {'label': 'Petrol', 'value': 'Petrol'},
                                    {'label': 'Diesel', 'value': 'Diesel'},
                                    {'label': 'Electric', 'value':'Electric'}
                                ],
                                value='Petrol',
                                style={'fontSize': '16px', 'borderRadius': '10px'}
                            )
                        ], style={'marginBottom': '25px'}),

                        # Owner Count
                        html.Div([
                            html.Label("Previous Owners", style={'fontWeight': '600', 'color': '#1E1E1E', 'marginBottom': '8px', 'display': 'block'}),
                            dcc.Input(
                                id='input5',
                                type='number',
                                placeholder='e.g., 1',
                                value=1,
                                min=1,
                                max=5,
                                style={
                                    'width': '100%',
                                    'padding': '15px 18px',
                                    'border': '2px solid #EDEDED',
                                    'borderRadius': '10px',
                                    'fontSize': '16px',
                                    'backgroundColor': '#FFFFFF'
                                }
                            )
                        ], style={'marginBottom': '25px'})
                        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                    # Right Column
                    html.Div([
                        # Brand
                        html.Div([
                            html.Label("Brand", style={'fontWeight': '600', 'color': '#1E1E1E', 'marginBottom': '8px', 'display': 'block'}),
                            dcc.Dropdown(
                                id='input6',
                                options=[
                                    {'label': 'Chevrolet', 'value': 'Chevrolet'},
                                    {'label': 'Honda', 'value': 'Honda'},
                                    {'label': 'BMW', 'value': 'BMW'},
                                    {'label': 'Hyundai', 'value': 'Hyundai'},
                                    {'label': 'Nissan', 'value': 'Nissan'},
                                    {'label': 'Tesla', 'value': 'Tesla'},
                                    {'label': 'Toyota', 'value': 'Toyota'},
                                    {'label': 'Kia', 'value': 'Kia'},
                                    {'label': 'Volkswagen', 'value': 'Volkswagen'},
                                    {'label': 'Ford', 'value': 'Ford'}
                                ],
                                value='Chevrolet',
                                style={'fontSize': '16px', 'borderRadius': '10px'}
                            )
                        ], style={'marginBottom': '25px'}),

                        # Transmission
                        html.Div([
                            html.Label("Transmission", style={'fontWeight': '600', 'color': '#1E1E1E', 'marginBottom': '8px', 'display': 'block'}),
                            dcc.Dropdown(
                                id='input7',
                                options=[
                                    {'label': 'Manual', 'value': 'Manual'},
                                    {'label': 'Automatic', 'value': 'Automatic'}
                                ],
                                value='Manual',
                                style={'fontSize': '16px', 'borderRadius': '10px'}
                            )
                        ], style={'marginBottom': '25px'}),
                        
                        # Color
                        html.Div([
                            html.Label("Color", style={'fontWeight': '600', 'color': '#1E1E1E', 'marginBottom': '8px', 'display': 'block'}),
                            dcc.Dropdown(
                                id='input8',
                                options=[
                                    {'label': 'White', 'value': 'White'},
                                    {'label': 'Silver', 'value': 'Silver'},
                                    {'label': 'Black', 'value': 'Black'},
                                    {'label': 'Red', 'value': 'Red'},
                                    {'label': 'Blue', 'value': 'Blue'},
                                    {'label': 'Gray', 'value': 'Gray'}
                                ],
                                value='White',
                                style={'fontSize': '16px', 'borderRadius': '10px'}
                            )
                        ], style={'marginBottom': '25px'}),

                    # Accidents
                    html.Div([
                        html.Label("Accidents Reported", style={'fontWeight': '600', 'color': '#1E1E1E', 'marginBottom': '8px', 'display': 'block'}),
                        dcc.Input(
                            id='input9',
                            type='number',
                            placeholder='e.g., 0',
                            value=0,
                            min=0,
                            max=5,
                            style={
                                    'width': '100%',
                                    'padding': '15px 18px',
                                    'border': '2px solid #EDEDED',
                                    'borderRadius': '10px',
                                    'fontSize': '16px',
                                    'backgroundColor': '#FFFFFF'
                                }
                            )
                        ], style={'marginBottom': '25px'}),                    

                        # Service History
                        html.Div([
                            html.Label("Service History", style={'fontWeight': '600', 'color': '#1E1E1E', 'marginBottom': '8px', 'display': 'block'}),
                            dcc.Dropdown(
                                id='input10',
                                options=[
                                    {'label': 'Full', 'value': 'Full'},
                                    {'label': 'Partial', 'value': 'Partial'}
                                ],
                                value='Full',
                                style={'fontSize': '16px', 'borderRadius': '10px'}
                            )
                        ], style={'marginBottom': '25px'}),
                        
                        # Insurance
                        html.Div([
                            html.Label("Insurance Valid", style={'fontWeight': '600', 'color': '#1E1E1E', 'marginBottom': '8px', 'display': 'block'}),
                            dcc.Dropdown(
                                id='input11',
                                options=[
                                    {'label': 'Yes', 'value': 'Yes'},
                                    {'label': 'No', 'value': 'No'}
                                ],
                                value='Yes',
                                style={'fontSize': '16px', 'borderRadius': '10px'}
                            )
                        ], style={'marginBottom': '25px'})
                    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
                ]),
                
                # Predict Button
                html.Div([
                    html.Button(
                        'Get Price Estimate',
                        id='predict-button',
                        n_clicks=0,
                        style={
                            'backgroundColor': '#007BFF',  # Vibrant Blue
                            'color': '#FFFFFF',
                            'border': 'none',
                            'padding': '18px 45px',
                            'fontSize': '18px',
                            'fontWeight': '600',
                            'borderRadius': '50px',
                            'cursor': 'pointer',
                            'transition': 'all 0.3s ease',
                            'boxShadow': '0 6px 20px rgba(0, 123, 255, 0.3)',
                            'width': '100%',
                            'maxWidth': '350px',
                            'textTransform': 'uppercase',
                            'letterSpacing': '1px'
                        }
                    )
                ], style={'textAlign': 'center', 'marginTop': '40px'}),
                
                # Confidence Note
                html.Div([
                    html.P("Your data is secure and private. This estimate is based on historical and current market data. Actual sale price may vary.", 
                           style={'color': '#1E1E1E', 'fontSize': '0.95rem', 'fontStyle': 'italic', 'textAlign': 'center', 'marginTop': '20px'})
                ])
            ], style={
                'backgroundColor': '#FFFFFF',
                'padding': '50px',
                'borderRadius': '20px',
                'boxShadow': '0 10px 40px rgba(0,0,0,0.1)',
                'marginBottom': '60px'
            }),
            
            # Results Section
            html.Div([
                html.Div(id='prediction-output', style={'textAlign': 'center'})
            ])
        ], id='prediction-section', style={'marginBottom': '80px'}),
        
        # Analytics Section
        html.Div([
            html.H2("Market Analytics & Insights", 
                   style={'color': '#1E1E1E', 'textAlign': 'center', 'marginBottom': '50px', 'fontSize': '2.5rem', 'fontWeight': '600'}),
            
            # Analytics Charts
            html.Div([
                html.Div([
                    dcc.Graph(id='brand-price-chart')
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='brand-popularity-chart')
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'})
            ]),
            
            html.Div([
                dcc.Graph(id='depreciation-chart')
            ], style={'marginTop': '30px'}),
            
            html.Div([
                dcc.Graph(id='user-input-analysis')
            ], style={'marginTop': '30px'})
        ], id='analytics-section', style={
            'backgroundColor': '#EDEDED',
            'padding': '80px 20px',
            'marginBottom': '0px'
        }),
        
        # Why Trust Section
        html.Div([
            html.H2("Why Trust Our Price Estimates?", 
                    style={
                        'color': '#1E1E1E', 
                        'textAlign': 'center', 
                        'marginBottom': '50px', 
                        'fontSize': '2.5rem', 
                        'fontWeight': '600'
                    }),
            html.Div([
                html.Div([
                    html.Div("🔍", style={'fontSize': '2rem', 'marginBottom': '20px'}),
                    html.H4("Precision Analytics", style={'color': '#1E1E1E', 'marginBottom': '15px', 'fontSize': '1.3rem'}),
                    html.P("Advanced ML algorithms trained on 50,000+ real car sales transactions", 
                           style={'color': '#1E1E1E', 'lineHeight': '1.6', 'fontSize': '1rem'})
                ], style={
                    'textAlign': 'center', 
                    'padding': '40px 30px', 
                    'backgroundColor': '#FFFFFF', 
                    'borderRadius': '15px', 
                    'boxShadow': '0 8px 25px rgba(0,0,0,0.1)', 
                    'width': '22%', 
                    'transition': 'transform 0.3s ease'
                }),
                html.Div([
                    html.Div("🔄", style={'fontSize': '2rem', 'marginBottom': '20px'}),
                    html.H4("Real-Time Updates", style={'color': '#1E1E1E', 'marginBottom': '15px', 'fontSize': '1.3rem'}),
                    html.P("Market data updated daily to reflect current trends and pricing patterns", 
                           style={'color': '#1E1E1E', 'lineHeight': '1.6', 'fontSize': '1rem'})
                ], style={
                    'textAlign': 'center', 
                    'padding': '40px 30px', 
                    'backgroundColor': '#FFFFFF', 
                    'borderRadius': '15px', 
                    'boxShadow': '0 8px 25px rgba(0,0,0,0.1)', 
                    'width': '22%', 
                    'transition': 'transform 0.3s ease'
                }),
                html.Div([
                    html.Div("⚖️", style={'fontSize': '2rem', 'marginBottom': '20px'}),
                    html.H4("Fair Market Value", style={'color': '#1E1E1E', 'marginBottom': '15px', 'fontSize': '1.3rem'}),
                    html.P("Prevents overpaying or underselling with transparent, unbiased pricing", 
                           style={'color': '#1E1E1E', 'lineHeight': '1.6', 'fontSize': '1rem'})
                ], style={
                    'textAlign': 'center', 
                    'padding': '40px 30px', 
                    'backgroundColor': '#FFFFFF', 
                    'borderRadius': '15px', 
                    'boxShadow': '0 8px 25px rgba(0,0,0,0.1)', 
                    'width': '22%', 
                    'transition': 'transform 0.3s ease'
                }),
                html.Div([
                    html.Div("⚡", style={'fontSize': '2rem', 'marginBottom': '20px'}),
                    html.H4("Instant & Secure", style={'color': '#1E1E1E', 'marginBottom': '15px', 'fontSize': '1.3rem'}),
                    html.P("Lightning-fast results with enterprise-grade security and privacy", 
                           style={'color': '#1E1E1E', 'lineHeight': '1.6', 'fontSize': '1rem'})
                ], style={
                    'textAlign': 'center', 
                    'padding': '40px 30px', 
                    'backgroundColor': '#FFFFFF', 
                    'borderRadius': '15px', 
                    'boxShadow': '0 8px 25px rgba(0,0,0,0.1)', 
                    'width': '22%', 
                    'transition': 'transform 0.3s ease'
                })
            ], style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'flexWrap': 'nowrap',
                'gap': '2%'
            })
        ], style={
            'backgroundColor': '#FFFFFF',
            'padding': '80px 20px',
            'marginBottom': '0px',
            'maxWidth': '1200px',
            'margin': '0 auto'
        }),
        
        # Contact Section
        html.Div([
            html.H2("Get in Touch with AfriCarModel", 
                   style={'color': '#FFFFFF', 'textAlign': 'left', 'marginBottom': '10px', 'fontSize': '1.5rem', 'fontWeight': '600', 'lineHeight': '0.2'}),
            
            html.Div([
                # Contact Info
                html.Div([
                    html.H3("Contact Information", style={'color': '#FFFFFF', 'marginBottom': '30px', 'fontSize': '1.0rem', 'marginTop': '30px'}),
                    html.Div([
                        html.Div("✉️", style={'fontSize': '1.5rem', 'marginRight': '15px', 'display': 'inline-block'}),
                        html.Span("support@africarsmodel.com", style={'fontSize': '1.2rem'})
                    ], style={'marginBottom': '20px', 'color': '#EDEDED'}),
                    html.Div([
                        html.Div("📞", style={'fontSize': '1.5rem', 'marginRight': '15px', 'display': 'inline-block'}),
                        html.Span("+250 788 123 456", style={'fontSize': '1.2rem'})
                    ], style={'marginBottom': '20px', 'color': '#EDEDED'}),
                    html.Div([
                        html.Div("📍", style={'fontSize': '1.5rem', 'marginRight': '15px', 'display': 'inline-block'}),
                        html.Span("Kigali, Rwanda", style={'fontSize': '1.2rem'})
                    ], style={'marginBottom': '30px', 'color': '#EDEDED'}),
                    
                    # Social Media
                    html.H4("Follow Us", style={'color': '#FFFFFF', 'marginBottom': '20px'}),
                    html.Div([
                        html.A("Twitter", href="https://twitter.com/carsmodelaf", target="_blank", 
                               style={'color': '#1DA1F2', 'textDecoration': 'none', 'margin': '0 20px 0 0', 'fontSize': '1.1rem', 'padding': '10px 15px', 'backgroundColor': 'rgba(255,255,255,0.1)', 'borderRadius': '25px', 'transition': 'all 0.3s ease'}),
                        html.A("Facebook", href="https://facebook.com/carsmodelaf", target="_blank",
                               style={'color': '#3B5998', 'textDecoration': 'none', 'margin': '0 20px 0 0', 'fontSize': '1.1rem', 'padding': '10px 15px', 'backgroundColor': 'rgba(255,255,255,0.1)', 'borderRadius': '25px'}),
                        html.A("LinkedIn", href="https://linkedin.com/company/carsmodel", target="_blank",
                               style={'color': '#0A66C2', 'textDecoration': 'none', 'margin': '0 20px 0 0', 'fontSize': '1.1rem', 'padding': '10px 15px', 'backgroundColor': 'rgba(255,255,255,0.1)', 'borderRadius': '25px'}),
                        html.A("Instagram", href="https://instagram.com/carsmodelaf", target="_blank",
                               style={'color': '#C13584', 'textDecoration': 'none', 'fontSize': '1.1rem', 'padding': '10px 15px', 'backgroundColor': 'rgba(255,255,255,0.1)', 'borderRadius': '25px'})
                    ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px'})
                ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Quick Contact Form
                html.Div([
                    html.H3("Quick Message", style={'color': '#FFFFFF', 'marginBottom': '25px', 'fontSize': '1.5rem', 'marginTop': '50px'}),
                    html.Div([
                        dcc.Input(
                            id='contact-name',
                            type='text',
                            placeholder='Your Name',
                            style={
                                'width': '80%',
                                'padding': '12px 15px',
                                'marginBottom': '15px',
                                'border': 'none',
                                'borderRadius': '8px',
                                'fontSize': '16px'
                            }
                        ),
                        dcc.Input(
                            id='contact-email',
                            type='email',
                            placeholder='Your Email',
                            style={
                                'width': '80%',
                                'padding': '12px 15px',
                                'marginBottom': '15px',
                                'border': 'none',
                                'borderRadius': '8px',
                                'fontSize': '16px'
                            }
                        ),
                        dcc.Textarea(
                            id='contact-message',
                            placeholder='Your Message',
                            style={
                                'width': '80%',
                                'padding': '12px 15px',
                                'marginBottom': '20px',
                                'border': 'none',
                                'borderRadius': '8px',
                                'fontSize': '16px',
                                'height': '100px',
                                'resize': 'vertical'
                            }
                        ),
                        html.Button(
                            'Send Message',
                            id='contact-submit',
                            style={
                                'backgroundColor': '#00B8A9',  # Teal
                                'color': '#FFFFFF',
                                'border': 'none',
                                'padding': '12px 30px',
                                'borderRadius': '25px',
                                'fontSize': '16px',
                                'fontWeight': '600',
                                'cursor': 'pointer',
                                'transition': 'all 0.3s ease'
                            }
                        )
                    ])
                ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '5%'})
            ])
        ], id='contact-section', style={
            'background': 'linear-gradient(135deg, #1A2E45 0%, #2B3A42 100%)',  # Muted Navy Gradient
            'padding': '80px 20px',
            'color': '#FFFFFF'
        })
    ], style={
        'maxWidth': '1200px',
        'margin': '0 auto',
        'padding': '0 20px'
    }),
    
    # Footer
    html.Footer([
        html.Div([
            html.P("© 2024 CarsModel. All rights reserved. | Powered by Advanced Machine Learning", 
                   style={'textAlign': 'center', 'margin': '0', 'color': '#FFFFFF'}),
            html.P("Transforming the African automotive market with intelligent pricing solutions.", 
                   style={'textAlign': 'center', 'margin': '10px 0 0 0', 'color': '#EDEDED', 'fontSize': '0.9rem'})
        ])
    ], style={
        'backgroundColor': '#1E1E1E',  # Dark Charcoal
        'padding': '30px 20px',
        'marginTop': '0px'
    })
], style={
    'backgroundColor': '#FFFFFF',
    'minHeight': '100vh',
    'fontFamily': '"Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    'margin': '0',
    'padding': '0'
})

# Enhanced callback with better error handling and formatting
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('input1', 'value'),  # make_year
    State('input2', 'value'),  # mileage_kmpl
    State('input3', 'value'),  # engine_cc
    State('input5', 'value'),  # owner_count 
    State('input9', 'value'),  # accidents_reported 
    State('input4', 'value'),  # fuel_type 
    State('input6', 'value'),  # brand
    State('input7', 'value'),  # transmission
    State('input8', 'value'),  # color
    State('input10', 'value'), # service_history
    State('input11', 'value')  # insurance_valid
)

def predict(n_clicks, make_year, mileage_kmpl, engine_cc, owner_count, accidents_reported,fuel_type,
            brand, transmission, color, service_history,insurance_valid):
    if n_clicks > 0:
        try:
            # Validate inputs
            inputs = [make_year, mileage_kmpl, engine_cc, owner_count, accidents_reported,fuel_type,
            brand, transmission, color, service_history,insurance_valid]
            
            if any(x is None for x in inputs):
                return html.Div([
                    html.H3("Please fill all fields", 
                           style={'color': '#FF6B6B', 'textAlign': 'center'})  # Coral
                ])

            current_year = datetime.now().year
            

            # Prepare input data
            columns = [
                'make_year', 'mileage_kmpl', 'engine_cc', 'owner_count', 'accidents_reported','fuel_type','brand',
                'transmission', 'color', 'service_history' , 'insurance_valid'
            ]

            # Create input DataFrame with explicit column mapping
            input_data = pd.DataFrame({
                'make_year': [make_year],
                'mileage_kmpl': [mileage_kmpl], 
                'engine_cc': [engine_cc],
                'owner_count': [owner_count],
                'accidents_reported': [accidents_reported],
                'fuel_type': [fuel_type],
                'brand': [brand],
                'transmission': [transmission],
                'color': [color],
                'service_history': [service_history],
                'insurance_valid': [insurance_valid]
            })

            # Make sure data types match your training data
            input_data['make_year'] = input_data['make_year'].astype('int64')
            input_data['mileage_kmpl'] = input_data['mileage_kmpl'].astype('float64')
            input_data['engine_cc'] = input_data['engine_cc'].astype('int64')
            input_data['owner_count'] = input_data['owner_count'].astype('int64')
            input_data['accidents_reported'] = input_data['accidents_reported'].astype('int64')
            
            # Ensure categorical columns are strings
            input_data['fuel_type'] = input_data['fuel_type'].astype('str')
            input_data['brand'] = input_data['brand'].astype('str')
            input_data['transmission'] = input_data['transmission'].astype('str')
            input_data['color'] = input_data['color'].astype('str')
            input_data['service_history'] = input_data['service_history'].astype('str')
            input_data['insurance_valid'] = input_data['insurance_valid'].astype('str')

            # Predict
            prediction = model_pipeline.predict(input_data)[0]
            
            # Format prediction
            formatted_price = f"${prediction:,.0f}"
            
            return html.Div([
                html.Div([
                    html.H2("Estimated Market Value", 
                           style={'color': '#1E1E1E', 'marginBottom': '20px', 'fontSize': '2rem'}),
                    html.H1(formatted_price, 
                           style={'color': '#00B8A9', 'fontSize': '3.5rem', 'fontWeight': '700', 'margin': '0', 'textShadow': '2px 2px 4px rgba(0,0,0,0.1)'}),  # Teal
                    html.P("Based on current market trends and vehicle specifications", 
                           style={'color': '#1E1E1E', 'fontSize': '1.2rem', 'margin': '15px 0 0 0'})
                ], style={
                    'backgroundColor': '#FFFFFF',
                    'padding': '50px',
                    'borderRadius': '20px',
                    'boxShadow': '0 10px 40px rgba(0,0,0,0.1)',
                    'textAlign': 'center',
                    'border': '4px solid #00B8A9',  # Teal
                    'marginBottom': '30px'
                }),
                
                # Additional insights
                html.Div([
                    html.H3("📊 Detailed Market Analysis", 
                           style={'color': '#1E1E1E', 'marginBottom': '30px', 'textAlign': 'center'}),
                    html.Div([
                        html.Div([
                            html.H4("Price Range", style={'fontWeight': '700', 'margin': '0 0 10px 0', 'color': '#1E1E1E'}),
                            html.P(f"${prediction*0.85:,.0f} - ${prediction*1.15:,.0f}", 
                                   style={'color': '#007BFF', 'fontSize': '1.4rem', 'margin': '0', 'fontWeight': '600'})  # Vibrant Blue
                        ], style={'textAlign': 'center', 'padding': '25px', 'backgroundColor': '#EDEDED', 'borderRadius': '15px', 'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}),
                        
                        html.Div([
                            html.H4("Age Factor", style={'fontWeight': '700', 'margin': '0 0 10px 0', 'color': '#1E1E1E'}),
                            html.P(f"{current_year - make_year} years old", 
                                   style={'color': '#FF6B6B', 'fontSize': '1.4rem', 'margin': '0', 'fontWeight': '600'})  # Coral
                        ], style={'textAlign': 'center', 'padding': '25px', 'backgroundColor': '#EDEDED', 'borderRadius': '15px', 'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'}),
                        
                        html.Div([
                            html.H4("Condition Score", style={'fontWeight': '700', 'margin': '0 0 10px 0', 'color': '#1E1E1E'}),
                            html.P(f"{max(1, 10 - accidents_reported - (owner_count*2))}/10", 
                                   style={'color': '#00B8A9', 'fontSize': '1.4rem', 'margin': '0', 'fontWeight': '600'})  # Teal
                        ], style={'textAlign': 'center', 'padding': '25px', 'backgroundColor': '#EDEDED', 'borderRadius': '15px', 'width': '30%', 'display': 'inline-block', 'margin': '0 1.5%'})
                    ]),
                    
                    # Market factors
                    html.Div([
                        html.H4("Key Factors Affecting Price:", style={'color': '#1E1E1E', 'marginTop': '30px', 'marginBottom': '15px'}),
                        html.Ul([
                            html.Li(f"Vehicle age: {current_year - make_year} years", style={'margin': '8px 0', 'color': '#1E1E1E'}),
                            html.Li(f"Ownership history: {owner_count} previous owner(s)", style={'margin': '8px 0', 'color': '#1E1E1E'}),
                            html.Li(f"Accident history: {accidents_reported} reported incident(s)", style={'margin': '8px 0', 'color': '#1E1E1E'}),
                            html.Li(f"Insurance status: {'Valid' if insurance_valid == 'Yes' else 'Expired'}", style={'margin': '8px 0', 'color': '#1E1E1E'})
                        ])
                    ], style={'textAlign': 'left', 'marginTop': '25px'})
                ], style={
                    'backgroundColor': '#FFFFFF',
                    'padding': '40px',
                    'borderRadius': '20px',
                    'boxShadow': '0 10px 40px rgba(0,0,0,0.1)'
                })
            ])
            
        except Exception as e:
            return html.Div([
                html.H3("Error in prediction", 
                       style={'color': '#FF6B6B', 'textAlign': 'center'}),  # Coral
                html.P(f"Please check your inputs and try again. Error:{str(e)}", 
                       style={'color': '#1E1E1E', 'textAlign': 'center'})
            ])

    return html.Div([
        html.Div([
            html.H3("Ready to Discover Your Car's Value?", 
                   style={'color': '#1E1E1E', 'textAlign': 'center', 'marginBottom': '20px'}),
            html.P("Fill in your vehicle details above and click 'Get Price Estimate' to discover your car's true market value using our advanced AI pricing engine.", 
                   style={'color': '#1E1E1E', 'textAlign': 'center', 'fontSize': '1.2rem', 'lineHeight': '1.6'})
        ], style={
            'backgroundColor': '#FFFFFF',
            'padding': '50px',
            'borderRadius': '20px',
            'boxShadow': '0 10px 40px rgba(0,0,0,0.1)',
            'textAlign': 'center'
        })
    ])

# Analytics callbacks
@app.callback(
    Output('brand-price-chart', 'figure'),
    Input('predict-button', 'n_clicks')
)
def update_brand_price_chart(n_clicks):
    # Calculate average price by brand from real data
    brand_avg_prices = car_price_df.groupby('brand')['price_usd'].mean().reset_index()
    brand_avg_prices = brand_avg_prices.sort_values('price_usd', ascending=False)
    
    fig = px.bar(
        brand_avg_prices, 
        x='brand', 
        y='price_usd',
        title='Average Price by Brand',
        color='price_usd',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        title_font_size=18,
        title_x=0.5,
        xaxis_title='Brand',
        yaxis_title='Average Price ($)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color='#1E1E1E'),
        showlegend=False
    )
    
    fig.update_traces(
        texttemplate='$%{y:,.0f}',
        textposition='outside'
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig

@app.callback(
    Output('brand-popularity-chart', 'figure'),
    Input('predict-button', 'n_clicks')
)
def update_brand_popularity_chart(n_clicks):
    # Calculate brand popularity based on count of listings
    brand_counts = car_price_df['brand'].value_counts().reset_index()
    brand_counts.columns = ['brand', 'count']
    
    fig = px.pie(
        brand_counts,
        values='count',
        names='brand',
        title='Brand Market Share (by listing count)',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        title_font_size=18,
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color='#1E1E1E')
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    return fig

@app.callback(
    Output('depreciation-chart', 'figure'),
    Input('predict-button', 'n_clicks')
)
def update_depreciation_chart(n_clicks):
    # Calculate average price by year from real data
    year_avg_prices = car_price_df.groupby('make_year')['price_usd'].mean().reset_index()
    year_avg_prices = year_avg_prices.sort_values('make_year')
    
    # Calculate depreciation relative to newest cars
    if not year_avg_prices.empty:
        max_price = year_avg_prices['price_usd'].max()
        year_avg_prices['value_retention'] = (year_avg_prices['price_usd'] / max_price) * 100
    else:
        # Fallback if no data
        year_avg_prices = pd.DataFrame({
            'make_year': list(range(2015, 2025)),
            'value_retention': [27, 30, 34, 39, 45, 52, 61, 72, 85, 100]
        })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=year_avg_prices['make_year'],
        y=year_avg_prices['value_retention'],
        mode='lines+markers',
        name='Vehicle Value',
        line=dict(color='#FF6B6B', width=4),  # Coral
        marker=dict(size=8, color='#E55A5A'),
        fill='tonexty'
    ))
    
    fig.update_layout(
        title='Vehicle Value Retention by Year',
        title_font_size=18,
        title_x=0.5,
        xaxis_title='Make Year',
        yaxis_title='Value Retention (%)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color='#1E1E1E'),
        hovermode='x unified'
    )
    
    return fig

@app.callback(
    Output('user-input-analysis', 'figure'),
    Input('predict-button', 'n_clicks'),
    State('input1', 'value'),
    State('input2', 'value'),
    State('input3', 'value'),
    State('input6', 'value')
)
def update_user_analysis(n_clicks, make_year, mileage_kmpl, engine_cc, brand):
    if n_clicks > 0 and all(v is not None for v in [make_year, mileage_kmpl, engine_cc, brand]):
        # Brand mapping
        brand_names = ['Chevrolet', 'Honda', 'BMW', 'Hyundai', 'Nissan', 'Tesla', 'Toyota', 'Kia', 'Volkswagen', 'Ford']
        selected_brand = brand if brand in brand_names else 'Unknown'
        
        # Calculate market averages from real data
        avg_year = car_price_df['make_year'].mean()
        avg_mileage = car_price_df['mileage_kmpl'].mean()
        avg_engine = car_price_df['engine_cc'].mean()
        
        # Get brand popularity from actual data
        brand_counts = car_price_df['brand'].value_counts(normalize=True) * 100
        brand_popularity = brand_counts.get(selected_brand, 10)
        
        categories = ['Age Factor', 'Mileage Efficiency', 'Engine Power', 'Brand Popularity']
        
        # Calculate scores (0-100)
        age_score = max(0, min(100, ((make_year - car_price_df['make_year'].min()) / 
                                   (car_price_df['make_year'].max() - car_price_df['make_year'].min())) * 100))
        
        mileage_score = max(0, min(100, (mileage_kmpl / car_price_df['mileage_kmpl'].max()) * 100))
        
        engine_score = max(0, min(100, ((engine_cc - car_price_df['engine_cc'].min()) / 
                                      (car_price_df['engine_cc'].max() - car_price_df['engine_cc'].min())) * 100))
        
        user_scores = [age_score, mileage_score, engine_score, brand_popularity]
        
        # Calculate market averages as percentile scores
        market_avg = [
            ((avg_year - car_price_df['make_year'].min()) / (car_price_df['make_year'].max() - car_price_df['make_year'].min())) * 100,
            (avg_mileage / car_price_df['mileage_kmpl'].max()) * 100,
            ((avg_engine - car_price_df['engine_cc'].min()) / (car_price_df['engine_cc'].max() - car_price_df['engine_cc'].min())) * 100,
            50
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=user_scores,
            theta=categories,
            fill='toself',
            name='Your Vehicle',
            line_color='#00B8A9'  # Teal
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=market_avg,
            theta=categories,
            fill='toself',
            name='Market Average',
            line_color='#007BFF'  # Vibrant Blue
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title=f'Your {selected_brand} vs Market Average',
            title_font_size=18,
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12, color='#1E1E1E')
        )
        
        return fig
    
    # Default empty chart
    fig = go.Figure()
    fig.update_layout(
        title='Vehicle Analysis (Enter details and predict to see analysis)',
        title_font_size=18,
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color='#1E1E1E')
    )
    return fig

# Run server
if __name__ == '__main__':
    app.run(debug=True)