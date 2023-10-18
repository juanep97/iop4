
# iop4lib config
import iop4lib
iop4conf = iop4lib.Config(config_db=False)


# django imports
from django.http import JsonResponse, HttpResponseBadRequest
from django.contrib.auth.decorators import permission_required

# iop4lib imports
from iop4lib.db import PhotoPolResult, AstroSource
from iop4lib.utils import get_column_values

# other imports
import os
import numpy as np
import pandas as pd
from astropy.time import Time

#logging
import logging
logger = logging.getLogger(__name__)


@permission_required(["iop4api.view_photopolresult", "iop4api.view_astrosource"])
def plot(request):

    source_name = request.POST.get("source_name", None)
    band = request.POST.get("band", "R")
    fmt = request.POST.get("fmt", "items") 
    date_start = request.POST.get("date_start", None)
    date_start = None if date_start == "" else date_start
    date_end = request.POST.get("date_end", None)
    date_end = None if date_end == "" else date_end

    enable_iop3 = request.POST.get("enable_iop3", False)
    enable_full_lc = request.POST.get("enable_full_lc", False)
    enable_errorbars = request.POST.get("enable_errorbars", False)


    # comment these lines to allow for empty plot
    if not AstroSource.objects.filter(name=source_name).exists(): 
        return HttpResponseBadRequest(f"Source '{source_name}' does not exist".format(source_name=source_name))

    # build plot

    import bokeh.colors
    from bokeh.document import Document
    from bokeh.transform import factor_cmap
    from bokeh.layouts import column, gridplot
    from bokeh.models import CategoricalColorMapper, LinearColorMapper, RangeTool, Range1d, LinearAxis, CustomJS, ColumnDataSource, Whisker, DatetimeAxis, DatetimeTickFormatter, Scatter, Segment, CDSView, GroupFilter, AllIndices, HoverTool, NumeralTickFormatter, BoxZoomTool, DataTable, TableColumn
    from bokeh.plotting import figure, show
    from bokeh.embed import components, json_item
    from bokeh.models import Div, Circle, Column, Row
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, Styles, InlineStyleSheet, GlobalInlineStyleSheet

    lod_threshold = 2000
    lod_factor = 10
    lod_timeout = 500

    # secondary axes values
    @np.vectorize
    def f_x1_to_x2(x1_val):
        return Time(x1_val, format='mjd').datetime
    
    # Get data from DB
    column_names = ['id', 'juliandate', 'band', 'instrument', 'mag', 'mag_err', 'p', 'p_err', 'chi', 'chi_err']
    qs = PhotoPolResult.objects.filter(astrosource__name=source_name, band=band).all()

    if date_start is not None:
        qs = qs.filter(juliandate__gte=Time(date_start).jd)

    if date_end is not None:
        qs = qs.filter(juliandate__lte=Time(date_end).jd)

    # iop3? 
    if enable_iop3:
        iop3_df = pd.read_csv(os.path.expanduser("~/iop3_results.csv"))
        iop3_idx = (iop3_df["name_IAU"] == source_name) & (iop3_df["filter"] == "R")
        if date_start is not None:
            iop3_idx = iop3_idx & (iop3_df["mjd_obs"] >= Time(date_start).mjd)
        if date_end is not None:
            iop3_idx = iop3_idx & (iop3_df["mjd_obs"] <= Time(date_end).mjd)
        iop3_df = iop3_df.loc[iop3_idx]
    
    # choose  the x and y
    if qs.count() > 0:
        vals = get_column_values(qs, column_names)
    else:
        vals = {k:np.array([]) for k in column_names}

    if enable_iop3:
        vals["instrument"] = np.append(vals["instrument"], list(map(lambda x: "IOP3-"+x, iop3_df["Telescope"])))
        vals["id"] = np.append(vals["id"], -np.arange(len(iop3_df)))
        vals["juliandate"] = np.append(vals["juliandate"], Time(iop3_df["mjd_obs"], format="mjd").jd)
        vals["mag"] = np.append(vals["mag"],  iop3_df['Mag'])
        vals["mag_err"] = np.append(vals["mag_err"], iop3_df['dMag'])
        vals["p"] = np.append(vals["p"], iop3_df['P']/100)
        vals["p_err"] = np.append(vals["p_err"], iop3_df['dP']/100)
        vals["chi"] = np.append(vals["chi"], iop3_df['Theta'])
        vals["chi_err"] = np.append(vals["chi_err"], iop3_df['dTheta'])

    if len(vals['id']) > 0:
        pks = vals['id']
        x1 = Time(vals['juliandate'], format='jd').mjd
        x2 = f_x1_to_x2(x1)
        y1 = vals['mag']
    else: # to allow empty plot
        vals = {k:np.array([]) for k in column_names}
        pks = np.array([])
        x1 = np.array([])
        x2 = np.array([])
        y1 = np.array([])

    # Instruments and colors (auto)
    instruments_L = sorted(list(set(vals['instrument'])), reverse=True)
    instrument_color_L = [bokeh.palettes.TolRainbow[max(3,len(instruments_L))][i % len(instruments_L)] for i in range(len(instruments_L))]

    rgb_L = [bokeh.colors.RGB.from_hex_string(hexcolor) for hexcolor in instrument_color_L]

    hsl_L_nonselected = [rgb.to_hsl() for rgb in rgb_L]
    hsl_L_selected= [rgb.to_hsl() for rgb in rgb_L]

    for hsl in hsl_L_nonselected:
        # make non-selected colors lighter and more gray
        hsl.l = 0.8
        hsl.s = 0.5

    for hsl in hsl_L_selected:
        hsl.l = 0.5
    
    palette_selected = [bokeh.colors.RGB.from_hsl(hsl) for hsl in hsl_L_selected]
    palette_nonselected = [bokeh.colors.RGB.from_hsl(hsl) for hsl in hsl_L_nonselected]
    
    index_cmap_selected = factor_cmap('instrument', 
                             palette=palette_selected, 
                             factors=instruments_L)
    
    index_cmap_nonselected = factor_cmap('instrument', 
                                   palette=palette_nonselected, 
                                   factors=instruments_L)
    
    source = ColumnDataSource(data=dict(pk = pks,
                                        instrument = vals['instrument'],
                                        x1 = x1, 
                                        x2 = x2, 
                                        y1 = vals['mag'], 
                                        y1_min = vals['mag']-vals['mag_err'],
                                        y1_max = vals['mag']+vals['mag_err'],
                                        y2 = vals['p'], 
                                        y2_min = vals['p']-vals['p_err'],
                                        y2_max = vals['p']+vals['p_err'],
                                        y3 = vals['chi'], 
                                        y3_min = vals['chi']-vals['chi_err'],
                                        y3_max = vals['chi']+vals['chi_err']),
                                name="source")

    view = CDSView(filter=AllIndices(), name="plot_view")

    # These will set the correspondence between axes x1 units and axes x2 units 

    if len(x1) >= 2:
        # for the main plot
        x1_lims = min(x1)-0.2*(max(x1)-min(x1)), max(x1)+0.2*(max(x1)-min(x1))
        x2_lims = f_x1_to_x2(x1_lims)

        # and for the subplots
        x1_range = min(x1)-0.05*(max(x1)-min(x1)), max(x1)+0.05*(max(x1)-min(x1))
        x2_range = f_x1_to_x2(x1_range)
    elif len(x1) == 1:
        x1_lims = x1[0]-0.2, x1[0]+0.2
        x2_lims = f_x1_to_x2(x1_lims)

        x1_range = x1[0]-0.05, x1[0]+0.05
        x2_range = f_x1_to_x2(x1_range)
    else:
        x1_lims = np.nan, np.nan
        x2_lims = np.nan, np.nan
        x1_range = np.nan, np.nan
        x2_range = np.nan, np.nan

    # also the freeze y axis range of the main plot
    if len(y1) >= 2:
        y1_lims = np.nanmin(y1)-0.05*(np.nanmax(y1)-np.nanmin(y1)), np.nanmax(y1)+0.05*(np.nanmax(y1)-np.nanmin(y1))
    elif len(y1) == 1:
        y1_lims = y1[0]-0.05, y1[0]+0.05
    else:
        y1_lims = np.nan, np.nan

    # Create RangeTool and Selected_range
    selected_range = Range1d(*x1_range)

    if enable_full_lc:
        range_tool = RangeTool(x_range=selected_range)
        range_tool.overlay.fill_color = "navy"
        range_tool.overlay.fill_alpha = 0.2

    # Create a hover tool with custom JS callback
    hover = HoverTool(renderers=[])
    hover.callback = CustomJS(args = dict(source = source, hover = hover), code = '''
                if (cb_data.index.indices.length > 0) { 
                    let index = cb_data.index.indices[0];
                    let p = source.data.y2[index];

                    if (isFinite(p)) {
                        hover.tooltips = [["id", "@pk"], ["instrument", "@instrument"], ["mag", "@y1"], ["p", "@y2{0.0 %}"], ["chi", "@y3"]];                                       
                    } else {
                        hover.tooltips = [["id", "@pk"], ["instrument", "@instrument"], ["mag", "@y1"]];
                    }
                } ''')
    
    # other tools

    #tools = ["fullscreen", "reset", "save", "pan", "auto_box_zoom", "wheel_zoom", "box_select", "lasso_select"]
    tools = ["fullscreen", "reset", "save", "pan", "wheel_zoom", "box_select", "lasso_select"]
    box_zoom_tool = BoxZoomTool(dimensions="auto")

    if enable_full_lc:
        # Create the main plot with range slider and secondary x-axis, fixed styles of markers

        # alternatively: # figure(sizing_mode="stretch", width=800, height=200,
        p_main = figure(x_range=x1_lims, tools="", lod_threshold=lod_threshold, lod_factor=lod_factor, lod_timeout=lod_timeout, output_backend="webgl")  
        p_main.circle(x='x1', y='y1', source=source, view=view,
                    size=3, 
                    line_color=bokeh.colors.named.navy, 
                    fill_color=bokeh.colors.named.navy, 
                    selection_line_color=bokeh.colors.named.navy.darken(0.1), 
                    nonselection_line_color=bokeh.colors.named.navy.lighten(0.55), 
                    selection_fill_color=bokeh.colors.named.navy.darken(0.1), 
                    nonselection_fill_color=bokeh.colors.named.navy.lighten(0.55), 
                    selection_fill_alpha=0.5, 
                    nonselection_fill_alpha=0.5, 
                    selection_line_alpha=0.5, 
                    nonselection_line_alpha=0.5)

        p_main.y_range = Range1d(*y1_lims)

        p_main.extra_x_ranges["secondary"] = Range1d(*x2_lims)
        p_main_ax_x2 = DatetimeAxis(x_range_name="secondary", axis_label="Date")
        p_main_ax_x2.formatter = DatetimeTickFormatter(months=r"%Y/%m/%d",
                                                days=r"%Y/%m/%d %H:%M",
                                                hours=r"%Y/%m/%d %H:%M",
                                                minutes=r"%Y/%m/%d %H:%M")
        p_main.add_layout(p_main_ax_x2, 'above')

        p_main.add_tools(range_tool)

    # Create three subplots that show data in the selected range tool of the main plot

    pltDict =   {   
                    'ax1': {
                                'x':"x1", 
                                'y':"y1", 
                                "y_label": "mag",
                                "err": ["y1_min", "y1_max"],                                  
                                "marker":"circle",
                                "size": 5,
                                "color": index_cmap_selected,
                                "selected_color": index_cmap_selected,
                                "nonselected_color": index_cmap_nonselected,
                                "alpha": 0.5,
                                "selected_alpha": 0.5,
                                "nonselected_alpha": 0.5,
                            }, 
                    'ax2':  {
                                'x':"x1", 
                                'y':"y2", 
                                "y_label": "p [%]",
                                "err": ["y2_min", "y2_max"],
                                "marker":"circle",
                                "size": 5,
                                "color": index_cmap_selected,
                                "selected_color": index_cmap_selected,
                                "nonselected_color": index_cmap_nonselected,
                                "alpha": 0.5,
                                "selected_alpha": 0.5,
                                "nonselected_alpha": 0.5,
                            },
                    'ax3':  {
                                'x':"x1", 
                                'y':"y3", 
                                "y_label": "chi [º]",
                                "err": ["y3_min", "y3_max"],                                  
                                "marker":"circle",
                                "size": 5,
                                "color": index_cmap_selected,
                                "selected_color": index_cmap_selected,
                                "nonselected_color": index_cmap_nonselected,
                                "alpha": 0.5,
                                "selected_alpha": 0.5,
                                "nonselected_alpha": 0.5,
                            },            
                }
                            
    for axLabel, axDict in pltDict.items():

        p = figure(title=None, x_range=selected_range, toolbar_location=None, tools=tools, lod_threshold=lod_threshold, lod_factor=lod_factor, lod_timeout=lod_timeout, output_backend="webgl")

        scatter_initial = Scatter(x=axDict['x'], y=axDict['y'], size=axDict["size"], fill_color=axDict["color"], line_color=axDict["color"], marker=axDict["marker"], fill_alpha=axDict["alpha"], line_alpha=axDict["alpha"])
        # scatter_selected = Scatter(x=axDict['x'], y=axDict['y'], size=axDict["size"], fill_color=axDict["selected_color"], line_color=axDict["selected_color"], marker=axDict["marker"], fill_alpha=axDict["selected_alpha"], line_alpha=axDict["selected_alpha"])
        scatter_nonselected = Scatter(x=axDict['x'], y=axDict['y'], size=axDict["size"], fill_color=axDict["nonselected_color"], line_color=axDict["nonselected_color"], marker=axDict["marker"], fill_alpha=axDict["nonselected_alpha"], line_alpha=axDict["nonselected_alpha"])
        pt_renderers = p.add_glyph(source, scatter_initial, selection_glyph=scatter_initial, nonselection_glyph=scatter_nonselected, view=view, name=f"{axLabel}_scatter_renderer")

        if enable_errorbars and axDict["err"] is not None:
            segs_initial = Segment(x0=axDict['x'], y0=axDict["err"][0], x1=axDict['x'], y1=axDict["err"][1], line_color=axDict["color"], line_alpha=axDict["alpha"])
            # segs_selected = Segment(x0=axDict['x'], y0=axDict["err"][0], x1=axDict['x'], y1=axDict["err"][1], line_color=axDict["selected_color"], line_alpha=axDict["selected_alpha"])
            segs_nonselected = Segment(x0=axDict['x'], y0=axDict["err"][0], x1=axDict['x'], y1=axDict["err"][1], line_color=axDict["nonselected_color"], line_alpha=axDict["nonselected_alpha"])
            errorbars_renderer = p.add_glyph(source, segs_initial, selection_glyph=segs_initial, nonselection_glyph=segs_nonselected, view=view, name=f"{axLabel}_errorbars_renderer")
            errorbars_renderer.visible = True

        if axLabel == "ax1":
            p.extra_x_ranges["secondary"] = Range1d(*x2_range)
            p_ax_x2 = DatetimeAxis(x_range_name="secondary")
            p_ax_x2.formatter = DatetimeTickFormatter(months=r"%Y/%m/%d",
                                                days=r"%Y/%m/%d %H:%M",
                                                hours=r"%Y/%m/%d %H:%M",
                                                minutes=r"%Y/%m/%d %H:%M")
            p.add_layout(p_ax_x2, 'above')

        if axLabel == "ax3":
            p.below[0].axis_label = "MJD"

        if axLabel != "ax3":
            # clear x-axis of subplots:
            # p.below = [] # or pass x_axis_location="below" or x_axis_location=None to figure constructor
            # or leave the axis but remove markings:
            p.below[0].major_tick_line_color = None  # turn off x-axis major ticks
            p.below[0].minor_tick_line_color = None  # turn off x-axis minor ticks
            p.below[0].major_label_text_font_size = '0pt'  # turn off x-axis tick labels

        if axLabel == "ax2":
            p.yaxis.formatter = NumeralTickFormatter(format=' 0.0 %')
            #p.y_range.start = -0.01

        p.yaxis.axis_label = axDict["y_label"]
        p.title = axDict["y_label"]
        p.title.visible = False

        # Add common hover tool with custom JS callback
        hover.renderers += [pt_renderers] # it has no renderers, append the scatter
        p.add_tools(hover)

        # make box_zoom the default active tool
        p.add_tools(box_zoom_tool)
        p.toolbar.active_drag = box_zoom_tool

        axDict["p"] = p

    # Calculate the proportionality factor and shift between primary and secondary axes limits from the initial ranges
    # link the secondary and primary axes ranges in the p1 subplot
    # because the range_tool changes the p1 primary axes but not the secondary (otherwise they are already linked)

    p1, p2, p3 = pltDict["ax1"]["p"], pltDict["ax2"]["p"], pltDict["ax3"]["p"]
    
    factor = (x2_lims[1] - x2_lims[0]) / (x1_lims[1] - x1_lims[0])
    shift = x2_lims[0] - x1_lims[0]*factor

    callback = CustomJS(args=dict(primary=p1.x_range, secondary=p1.extra_x_ranges["secondary"], factor=factor, shift=shift), code = """secondary.start = factor * primary.start + shift; secondary.end = factor * primary.end + shift;""")
    p1.x_range.js_on_change('start', callback)
    p1.x_range.js_on_change('end', callback)

    # Combine plots in a column layout

    if enable_full_lc:
        plot_layout = gridplot([[p_main], [pltDict["ax1"]["p"]], [pltDict["ax2"]["p"]], [pltDict["ax3"]["p"]]], merge_tools=True, toolbar_location="right", sizing_mode='stretch_both')
    else:
        plot_layout = gridplot([[pltDict["ax1"]["p"]], [pltDict["ax2"]["p"]], [pltDict["ax3"]["p"]]], merge_tools=True, toolbar_location="right", sizing_mode='stretch_both')

    ##############
    # Data table #
    ##############
    from bokeh.models.widgets import HTMLTemplateFormatter
    table_link_formatter = HTMLTemplateFormatter(template="""<a href="/iop4/admin/iop4api/photopolresult/?id=<%= value %>"><%= value %></a>""")
    table_columns = [   TableColumn(field="pk", title="id", formatter=table_link_formatter),
                        TableColumn(field="instrument", title="instrument"),
                        TableColumn(field="x1", title="mjd"),
                        TableColumn(field="y1", title="mag"),
                        TableColumn(field="y1_err", title="dmag"),
                        TableColumn(field="y2", title="p"),
                        TableColumn(field="y2_err", title="dp"),
                        TableColumn(field="y3", title="chi"),
                        TableColumn(field="y3_err", title="dchi")]
               
    data_table = DataTable(source=source, view=view, columns=table_columns, index_position=None,
                           sortable=True, reorderable=True, scroll_to_selection=True, 
                           stylesheets=[InlineStyleSheet(css=""".slick-cell.selected { background-color: #d2eaff; }""")])

    # Add a callback to hide errorbars when panning (it makes the plot smoother)

    from bokeh.events import Event, PanStart, PanEnd, Press, PressUp, RangesUpdate

    if enable_errorbars:
        cb_hide_errorbars = """window.were_errorbars_active = document.querySelector('#cbox_errorbars').checked; plot_hide_errorbars();"""
        cb_restore_errorbars = """if (window.were_errorbars_active) {plot_show_errorbars();}"""

        if enable_full_lc:
            p_main.js_on_event(PanStart, CustomJS(code=cb_hide_errorbars))
            p_main.js_on_event(PanEnd, CustomJS(code=cb_restore_errorbars))

        p1.js_on_event(PanStart, CustomJS(code=cb_hide_errorbars))
        p1.js_on_event(PanEnd, CustomJS(code=cb_restore_errorbars))
        p2.js_on_event(PanStart, CustomJS(code=cb_hide_errorbars))
        p2.js_on_event(PanEnd, CustomJS(code=cb_restore_errorbars))
        p3.js_on_event(PanStart, CustomJS(code=cb_hide_errorbars))
        p3.js_on_event(PanEnd, CustomJS(code=cb_restore_errorbars))

    #################################################
    # Create a legend (it will be a different plot) #
    #################################################

    # Create a ColumnDataSource with data for Circle and Star glyphs
    legend_source = ColumnDataSource(data=dict(x=[0], y=[0]))

    label_stylesheet = InlineStyleSheet(css="""
                                        :host { 
                                            display: flex;
                                            align-items: self-end;
                                            margin: 0;
                                            margin-left: 10px;
                                            min-width: 8em;
                                            max-width: 12em;
                                        }

                                        label {
                                            display: block;
                                            cursor: pointer; 
                                            vertical-align: middle;
                                            font-weight: normal;
                                        }

                                        span {
                                            display: inline-block;
                                            min-width: 8em;
                                        }

                                        input {
                                            cursor: pointer;
                                        }

                                        label:active {
                                            background-color: #f2f2f2;
                                            border-radius: 4px;
                                        }
                                        """)

    legend_row_L = list()
    for instrument, color in zip(instruments_L, instrument_color_L):
        # Create Div elements for labels
        label = Div(text=f"""<label><span>{instrument}</span><input onclick="plot_hide_instrument(this);" data-instrument="{instrument}" type="checkbox" checked/></label>""", height=21, stylesheets=[label_stylesheet])
        # Create glyphs
        circle_glyph = Circle(x='x', y='y', fill_color=color, line_color=color, size=8)
        # Create plots to hold the glyphs
        legend_p = figure(width=21, height=21, toolbar_location=None, min_border=0, tools="")
        legend_p.add_glyph(legend_source, circle_glyph)
        legend_p.axis.visible = False
        legend_p.grid.visible = False
        legend_p.background_fill_alpha = 0.0
        legend_p.outline_line_alpha = 0.0

        # Combine glyphs and labels into rows
        legend_row = Row(children=[legend_p, label], align="center")

        legend_row_L.append(legend_row)

    # Combine rows into a column and display
    legend_layout = Column(children=legend_row_L)




    # Get the components to embed in the Django template

    doc = Document()
    doc.add_root(plot_layout)
    doc.add_root(data_table)
    doc.add_root(legend_layout)
    
    if fmt == "components":
        # these can be loaded independently in the template and will be connected, but no control on the load process
        script, (div_plot, div_table, div_legend) = components([plot_layout, data_table, legend_layout], wrap_script=False)
        return JsonResponse({
                                'plot': {
                                        'script': script,
                                        'div_plot': div_plot,
                                        'div_table': div_table,
                                        'div_legend': div_legend,
                                    }, 
                                'n_points':len(x1),
                                'enable_full_lc': enable_full_lc,
                                'enable_iop3': enable_iop3,
                                'enable_errorbars': enable_errorbars,
                            })
    
    elif fmt == "json_item":
        # this will make them indepenedent, so table wont be connected to plot
        return JsonResponse({'plot':json_item(plot_layout), 
                             'legend':json_item(legend_layout),
                             'table':json_item(data_table),
                             'n_points':len(x1),
                             'enable_full_lc': enable_full_lc,
                             'enable_iop3': enable_iop3,
                             'enable_errorbars': enable_errorbars})
    
    elif fmt == "items":
        # to implement my own load process
        from bokeh.embed.util import standalone_docs_json_and_render_items
        doc, render_items = standalone_docs_json_and_render_items([plot_layout, data_table, legend_layout])
        return JsonResponse({'doc':doc,
                            'render_items':[item.to_json() for item in render_items],
                            'n_points':len(x1),
                            'enable_full_lc': enable_full_lc,
                            'enable_iop3': enable_iop3,
                            'enable_errorbars': enable_errorbars})
    
    else:

        return HttpResponseBadRequest("Invalid format parameter")

