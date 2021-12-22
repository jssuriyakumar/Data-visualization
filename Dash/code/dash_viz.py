
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dash_table.Format import Group
from dash.dependencies import Input, Output
import plotly.express as px
import dash_table

external_stylesheets = [dbc.themes.LUX]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv(r"C:\Users\sip011635\Documents\My files\Loan Eligibility Prediction\Data\train.csv",
                 usecols=['Gender','Married', 'Dependents', 'Education', 'Self_Employed',
       'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status'])


loan_val = pd.DataFrame(df['Loan_Amount_Term'].value_counts(sort='descending').head(3)).reset_index()
pa = df.groupby('Property_Area')
urb = pd.DataFrame(pa.get_group('Urban')['Loan_Status'].value_counts().reset_index())
rur = pd.DataFrame(pa.get_group('Rural')['Loan_Status'].value_counts().reset_index())
semiurb = pd.DataFrame(pa.get_group('Semiurban')['Loan_Status'].value_counts().reset_index())

mart = df.groupby(['Gender','Married'])['Gender','Married','Loan_Status']
b = round(pd.DataFrame(mart.get_group(('Male','Yes')).value_counts().reset_index())[0][0]/(pd.DataFrame(mart.get_group(('Male','Yes')).value_counts().reset_index())[0][0]+pd.DataFrame(mart.get_group(('Male','Yes')).value_counts().reset_index())[0][1])*100,2)
c = round(pd.DataFrame(mart.get_group(('Male','No')).value_counts().reset_index())[0][0]/(pd.DataFrame(mart.get_group(('Male','No')).value_counts().reset_index())[0][0]+pd.DataFrame(mart.get_group(('Male','Yes')).value_counts().reset_index())[0][1])*100,2)
d = round(pd.DataFrame(mart.get_group(('Female','Yes')).value_counts().reset_index())[0][0]/(pd.DataFrame(mart.get_group(('Female','Yes')).value_counts().reset_index())[0][0]+pd.DataFrame(mart.get_group(('Female','Yes')).value_counts().reset_index())[0][1])*100,2)
e = round(pd.DataFrame(mart.get_group(('Female','No')).value_counts().reset_index())[0][0]/(pd.DataFrame(mart.get_group(('Female','No')).value_counts().reset_index())[0][0]+pd.DataFrame(mart.get_group(('Female','No')).value_counts().reset_index())[0][1])*100,2)

df1 = pd.DataFrame(df.groupby(['Self_Employed','Married'])['Loan_Status'].count().reset_index())
df1.rename(columns = {'Loan_Status':'Loan_Application_Count'},inplace=True)


p= pd.DataFrame(df.isna().sum(),columns=['NA'])
p['NA Percent']=1.0;

for i in range(df.shape[1]):
    pp = round((df.iloc[:,i].isna().sum()/df.iloc[:,i].fillna(0).count())*100,2)
    p['NA Percent'][i] = pp

na = []
na_per = []
for i in p['NA']:
    na.append(i)
for j in p['NA Percent']:
    na_per.append(j)
na_table = pd.DataFrame(na,columns=['NA'])
na_table['NA Percent'] = na_per
na_table['Column_Name'] = df.columns
na_table = na_table[['Column_Name','NA','NA Percent']]


gml = df.groupby(['Gender','Married','Self_Employed'])['Loan_Status']


colors = {
    'background': '#111111',
    'text': '#7FDBFF',
    'text1': '#FFFFFF',
    'text2' : '#000000'
}

app.layout = html.Div([
        dbc.Row([
        dbc.Col(
            html.Img(
                src="https://cdn-icons-png.flaticon.com/128/2660/2660516.png",
                style = {'textAlign': 'right','height':'100px'}),width =1,),
        dbc.Col(        style={'backgroundColor': '#25597f', 'color': 'white'},

            children=[
                html.H1("LOAN ELIGIBILITY PREDICTION AT A GLANCE",
                        style={'color': 'white','font-size':'30px'}),
                html.P('Visualising trends across the loan approval and non-approval status of a private bank'),
                html.Hr(),
                # html.P("CREDITS: This page is completely built based on Plotly - Dash library. Thanks to Plotly.",
                #        style={'font-size':'10px'})
                
                ]),
        ]),
    
        dbc.Row([
            dbc.Row(dbc.Card(html.H3(children='Overview of the Loan Eligibility Dataset',
                                     className="text-center text-light bg-dark"), body=True, color="dark")
                    , className="mt-4 mb-5")
        ]),    
        


        dash_table.DataTable(
            id='datatable',
            columns=[{"name": i, "id": i} for i in df.columns],
        
            data=df.to_dict('records')[:6],
            
            style_table={'overflowX': 'scroll',
                            'padding': 10},
            style_header={'backgroundColor': '#25597f', 'color': 'white'},
            style_cell={
                'backgroundColor': 'white',
                'color': 'black',
                'fontSize': 13,
                'font-family': 'Nunito Sans'}),
        
        html.Br(),
        
        dbc.Row(dbc.Card(html.H3(children='Missing value percentage of each columns (Independent variables)',
                                 className="text-center text-light bg-dark"), body=False, color="dark")
                , className="mt-4 mb-5"),
        
        dash_table.DataTable(

            id='datatable1',
            
            columns=[{"name": i, "id": i} for i in na_table.columns],
        
            data=na_table.to_dict('records'),
            sort_action="native",
            sort_mode="multi",
            
            style_table={'overflowX': 'scroll',
                            'padding': 3},
            style_header={'backgroundColor': '#25597f', 'color': 'white'},
            style_cell={
                'backgroundColor': 'white',
                'color': 'black',
                'fontSize': 13,
                'font-family': 'Nunito Sans',
                'if': {'column_id': 'Column_Name'},'textAlign': 'left',
        }),
        html.Br(),
        
        
            dbc.Row(dbc.Card(html.H3(children='Data Proportion of different categorical variables',
                                     className="text-center text-light bg-dark"), body=False, color="dark")
                    , className="mt-4 mb-5"),
            # html.Br(),
            dcc.Dropdown(
                id = 'bar_dropdown',
                options = [
                    {'label':'Gender','value':'Gender'},
                    {'label':'Education','value':'Education'}, #, 'disabled': True
                    {'label':'Self_Employed','value':'Self_Employed'},
                    {'label':'Property_Area','value':'Property_Area'},
                    {'label':'Credit_History','value':'Credit_History'},
                    {'label':'Loan_Status','value':'Loan_Status'},
                    
                    ],
                value = 'Credit_History',
                multi = False,
                clearable = False,
                searchable = True,
                #placeholder="Select/Type a city",
                style = {"width" : "50%"}),
            
        html.Div([
            dcc.Graph(id ='bar_graph')
            ]),
        
        html.Br(),
        html.Br(),
        
        dbc.Row(dbc.Card(html.H3(children='Data Proportion of different categorical variables',
                                 className="text-center text-light bg-dark"), body=False, color="dark")
                , className="mt-4 mb-5"),
        
        html.Div([
            html.P("x-axis:"),
        dcc.Checklist(
            id = 'x_box_dropdown',
            options = [{'value': i, 'label':i} for i in ['Gender', 'Married', 'Dependents','Education','Self_Employed',]],
            value = ['Dependents'],
            labelStyle={'display': 'inline-block'}),
            
            html.P("y-axis:"),
        dcc.RadioItems(
            id='y_box_dropdown', 
            options=[{'value': x, 'label': x} 
                     for x in ['ApplicantIncome','CoapplicantIncome','LoanAmount']],
            value='LoanAmount', 
            labelStyle={'display': 'inline-block'}
    ),
        dcc.Graph(id="box-plot"),
        ]),
        
        
        dbc.Row(dbc.Card(html.H3(children='PDF and CDF of continous variables',
                                 className="text-center text-light bg-dark"), body=False, color="dark")
                , className="mt-4 mb-5"),
        
        html.Div([
            
        dcc.Dropdown(
            id = 'line_dropdown',
            options = [{'value': i, 'label':i} for i in ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']],
            value = 'ApplicantIncome',
            
            ),
            
            
    
        dcc.Graph(id="line_plot"),
        ]),

        
        dbc.Row(dbc.Card(html.H3('Loan approval rate across different property locations',
                                 className="text-center text-light bg-dark"), body=False, color="dark")
                , className="mt-4 mb-5"),
        
        dcc.Dropdown(
            id = 'propertyarea_dropdown',
            options = [
                {'label':'Property_Area','value':'Property_Area'}],
            
            value = 'Property_Area',
            searchable = False,
            style = {"width" : "50%"}),
        
        html.Div([
            dcc.Graph(id ='propertyarea_graph'),
            
            ]),
        
        html.Br(),
        html.Br(),
        
        dbc.Row(dbc.Card(html.H3('Influence of marital status on loan approval rate',
                                 className="text-center text-light bg-dark"), body=False, color="dark")
                , className="mt-4 mb-5"),
        
        dcc.Dropdown(
            id = 'marital_dropdown',
            options = [
                {'label':'Gender','value':'Gender'}],
            
            value = 'Gender',
            searchable = False,
            style = {"width" : "50%"}),
        
        html.Div([
            dcc.Graph(id ='marital_graph'),
            
            ]),
        
        html.Br(),
        html.Br(),
        
     
        
        
        dbc.Row(dbc.Card(html.H3('Correlation plot of continous variables',
                                 className="text-center text-light bg-dark"), body=False, color="dark")
                , className="mt-4 mb-5"),
        
        html.Div([
            dcc.ConfirmDialog(
                id = 'confirm_dialog',
                displayed= False,
                message = 'Please choose a Dropdown variable !'),
        
        dcc.Dropdown(
            id = 'scatter_dropdown',
            options = [{'label':s, 'value':s} for s in ['ApplicantIncome','CoapplicantIncome','LoanAmount']],
            value = ['ApplicantIncome','CoapplicantIncome','LoanAmount'],
            multi = True,
            searchable = False),
        
        dcc.Graph(id='scatter_graph',figure = {}),

            ]),
        html.Br(),
        html.Br(),
        
        dbc.Row(dbc.Card(html.H3(f"Out of {df['Loan_Amount_Term'].count()} applied loans {round((df['Loan_Amount_Term'].value_counts().reset_index()['Loan_Amount_Term'][0]/df['Loan_Amount_Term'].count()*100),2)}% loans have been applied for Loan Term for {df['Loan_Amount_Term'].value_counts().reset_index()['index'][0]} months",
                                 className="text-center text-light bg-dark"), body=False, color="dark")
                , className="mt-4 mb-5"),
        
        dcc.Dropdown(
            id = 'loanterm_dropdown',
            options = [
                {'label':'Loan_Amount_Term','value':'Loan_Amount_Term'}],
            
            value = 'Loan_Amount_Term',
            searchable = False,
            style = {"width" : "50%"}),
        
        html.Div([
            dcc.Graph(id ='bar_loan_graph'),
            
            ]),
        
        
        html.Br(),

        
        # dash_table.DataTable(

        #     id='datatable2',
            
        #     columns=[{"name": i, "id": i} for i in df1.columns],
        
        #     data=df1.to_dict('records'),
            
        #     style_table={'overflowX': 'scroll',
        #                     'padding': 3},
        #     style_header={'backgroundColor': '#25597f', 'color': 'white'},
        #     style_cell={
        #         'backgroundColor': 'white',
        #         'color': 'black',
        #         'fontSize': 13,
        #         'font-family': 'Nunito Sans'}),
        dbc.Row(dbc.Card(html.H3('Influence of Education, marital status, Employment type on loan approval rate',
                                 className="text-center text-light bg-dark"), body=False, color="dark")
                , className="mt-4 mb-5"),
        
        dcc.Dropdown(
            id = 'gem_dropdown',
            options = [
                {'label':'Gender','value':'Gender'}],
            
            value = 'Gender',
            searchable = False,
            style = {"width" : "50%"}),
        
        html.Div([
            dcc.Graph(id ='gem_graph'),
            
            ]),
        
        
        
        html.Br(),
        html.Br(),   
    
    ])


@app.callback(
    Output(component_id="bar_graph",component_property="figure"),
    [Input(component_id="bar_dropdown",component_property="value")],
    )

def update_bar_graph(bar_dropdown):
    
    df_copy = df.copy()
    
    pie_graph = px.pie(data_frame=df_copy,names=bar_dropdown,hole=.5)
    pie_graph.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,)

    return pie_graph

@app.callback(
    Output(component_id="box-plot",component_property="figure"),
    [Input("x_box_dropdown", "value"), Input("y_box_dropdown", "value")])

def update_box_graph(x,y):
    fig = px.box(df, x=x, y=y,points='all',notched=True)
    
    fig.update_layout(height=500,
                                width=800,)
    return fig




@app.callback(Output(component_id="bar_loan_graph",component_property="figure"),
              [Input(component_id="loanterm_dropdown",component_property="value")],)

def update_loan_bar_graph(loanterm_dropdown):
    bar_loan_term = go.Figure(data=[go.Bar(
    x=loan_val['index'].tolist(),
    y=loan_val['Loan_Amount_Term'].tolist(),
    width=[45, 35, 35],
    text=loan_val['Loan_Amount_Term'].tolist(),
    textposition='outside',
    name = 'Loan_Amount_Term'
    ),
        
    ])
    bar_loan_term.update_layout(height=500,
                                width=840,
                                title_text = f"Out of {df['Loan_Amount_Term'].count()} applied loans {round((df['Loan_Amount_Term'].value_counts().reset_index()['Loan_Amount_Term'][0]/df['Loan_Amount_Term'].count()*100),2)}% loans have been applied for Loan Term for {df['Loan_Amount_Term'].value_counts().reset_index()['index'][0]} months")
     
    return bar_loan_term

@app.callback(Output(component_id="propertyarea_graph",component_property="figure"),
              [Input(component_id="propertyarea_dropdown",component_property="value")],
    )

def update_propertyarea_graph(propertyarea_dropdown):
    fig = make_subplots(rows=1, cols=3,subplot_titles=('Urban','Rural','Semiurban'),)

    fig.add_trace(go.Bar(x = urb['index'],
                         y = urb['Loan_Status'],
                         
                         texttemplate = ['Loan Approved','Loan Not Approved'],
                         textposition="outside",
                         textfont_color="black",
                         name = 'Urban',
                         ),
                          
              row=1, col=1)
    
    fig.add_trace(go.Bar(x = rur['index'],
                         y = rur['Loan_Status'],
                         texttemplate = ['Loan Approved','Loan Not Approved'],
                         textposition="outside",
                         name = 'Rural'),
              row=1, col=2)
    
    fig.add_trace(go.Bar(x = semiurb['index'],
                         y = semiurb['Loan_Status'],
                         texttemplate = ['Loan Approved','Loan Not Approved'],
                         textposition="outside",
                         name = 'Semirural'),
              row=1, col=3)


    fig.update_layout(height=500, width=1300,
                  title_text = "Loan Approval rate based on Property Area",
                  )
    fig['layout']['xaxis']['title'] = f"Urban loan approval rate {round(urb['Loan_Status'][0]/urb['Loan_Status'][0]+urb['Loan_Status'][1],2)}%"
    fig['layout']['xaxis2']['title'] = f"Rural loan approval rate {round(rur['Loan_Status'][0]/rur['Loan_Status'][0]+rur['Loan_Status'][1],2)}%"
    fig['layout']['xaxis3']['title'] = f"Semiurban loan approval rate {round(semiurb['Loan_Status'][0]/semiurb['Loan_Status'][0]+semiurb['Loan_Status'][1],2)}%"
    
    return fig



@app.callback(Output(component_id="marital_graph",component_property="figure"),
              [Input(component_id="marital_dropdown",component_property="value")],
    )

def update_martial_graph(marital_dropdown):
    fig = make_subplots(rows=2, cols=2,subplot_titles=('Married Men','Unmarried Men','Married Women','Unmarried Women',),)
    k = 1
    for i in ['Male','Female']:  
        l = 1
        for j in ['Yes','No']:
            a = pd.DataFrame(mart.get_group((i,j)).value_counts().reset_index())
            fig.add_trace(go.Bar(x = a['Loan_Status'],
                                 y = a[0],
                                 
                                 texttemplate = ['Loan Approved','Loan Not Approved'],
                                 textposition="outside",
                                 textfont_color="black",
                                 # name = 'Urban',
                                 ),
                                  
                      row=k, col=l)
            l+=1
        k+=1
    fig['layout']['xaxis']['title'] = f"Loan approval rate for Married Men {b}%"
    fig['layout']['xaxis2']['title'] = f"Loan approval rate for Unmarried Men {c}%"
    fig['layout']['xaxis3']['title'] = f"Loan approval rate for Married Women {d}%"
    fig['layout']['xaxis4']['title'] = f"Loan approval rate for Unmarried Women {e}%"


    fig.update_layout(height=500, width=1300,
                  title_text = "Loan Approval rate based on Property Area",
                  )
    return fig

@app.callback(
    Output(component_id="line_plot",component_property="figure"),
    [Input(component_id='line_dropdown',component_property='value')])


def update_line_graph(line_dropdown):
    #for i in ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']:
    fig = go.Figure()
    counts, bin_edges = np.histogram(df[line_dropdown].dropna(), bins=10)
    pdf=counts/sum(counts)
    cdf=np.cumsum(pdf)
    fig.add_traces(data=go.Scatter(x = bin_edges[1:],y = pdf,name = 'pdf')) #px.line(x = np.linspace(0,1,10),y = cdf,markers="True")
    fig.add_traces(data=go.Scatter(x = bin_edges[1:],y = cdf,name = 'cdf'))
    return fig
   
   
@app.callback(
    Output(component_id="confirm_dialog",component_property="displayed"),
    Output(component_id="scatter_graph",component_property="figure"),
    [Input(component_id="scatter_dropdown",component_property="value")],
    )


def update_scatter_graph(continous):
    
    if len(continous)>0:
        df_scatter_copy = df.copy()
        scatter_chart = px.scatter_matrix(df_scatter_copy, dimensions=continous,
                                          color='Loan_Amount_Term')
        scatter_chart.update_traces(diagonal_visible = False, showupperhalf = False)
        
        return False, scatter_chart
    
    if len(continous) == 0:
        return True, dash.no_update
        
@app.callback(Output(component_id="gem_graph",component_property="figure"),
              [Input(component_id="gem_dropdown",component_property="value")],
    )

def update_gem_graph(gem_dropdown):
    figg = make_subplots(rows=4, cols=2,)
    l = 1
    for i in ['Male','Female']:  
        for j in ['Yes','No']:
            n = 1
            for k in ['Yes','No']:
                b = pd.DataFrame(gml.get_group((i,j,k)).value_counts().reset_index())
                figg.add_trace(go.Bar(x = ['Loan Approved','Loan Not Approved'],
                                     y = b['Loan_Status'].tolist(),
                                     
                                     texttemplate =  [b['Loan_Status'][0], b['Loan_Status'][1]],
                                     textposition="inside",
                                     textfont_color="black",
                                     name = f'{i}, Married {j} Employment {k}',
                                     ),
                                      
                          row=l, col=n)
            
                n+=1
            l+=1
    
    r = 0
    for i in ['Male','Female']:  
        for j in ['Yes','No']:
            for k in ['Yes','No']:
                b = pd.DataFrame(gml.get_group((i,j,k)).value_counts().reset_index())
                r+=1
                #print(r,i,j,k,round(b['Loan_Status'][0]/(b['Loan_Status'][0]+b['Loan_Status'][1]),2))
                figg['layout'][f'xaxis{r}']['title'] = f"Loan approval rate for Gender - {i} Married - {j} Self-Employed {k} is {round(b['Loan_Status'][0]/(b['Loan_Status'][0]+b['Loan_Status'][1]),2)}%"
                

            
            
    figg.update_layout(height=1000, width=1500,
                  title_text = "Loan Approval rate based on Marital status Education and Employmenrt",
                  )
    return figg




if __name__ == '__main__':
    app.run_server(debug=True)
