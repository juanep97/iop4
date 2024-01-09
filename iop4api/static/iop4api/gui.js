/* GUI */

document.addEventListener('DOMContentLoaded', function() {

    /* GUI NAVBAR shrink on scroll */ 

    window.addEventListener('scroll', function() {
        
        const navbar = document.getElementById('navbar');
        const appBody = document.getElementById('appBody');

        if (window.scrollY > 10) {
            document.querySelector("#navbar h2").style.marginLeft = getComputedStyle(document.documentElement).getPropertyValue('--navbar-gaps-scrolled');
            document.querySelector("#navbar h2").style.fontSize = getComputedStyle(document.documentElement).getPropertyValue('--navbar-h2-fontsize-scrolled');
            document.querySelector("#navbar").classList.add("scrolled");
            navbar.style.height = getComputedStyle(document.documentElement).getPropertyValue('--navbar-scrolled-height');
        } else {
            document.querySelector("#navbar h2").style.marginLeft = getComputedStyle(document.documentElement).getPropertyValue('--navbar-gaps-initial');
            document.querySelector("#navbar h2").style.fontSize = getComputedStyle(document.documentElement).getPropertyValue('--navbar-h2-fontsize-initial');
            document.querySelector("#navbar").classList.remove("scrolled");
            navbar.style.height = getComputedStyle(document.documentElement).getPropertyValue('--navbar-initial-height');
        }
    });

});


/* Utility Functions */

function getCurrentDateTime() {
    const now = new Date();

    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');  // JavaScript months are 0-indexed
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');

    return `${year}/${month}/${day} ${hours}:${minutes}`;
}

/*****************************/
/********** TABLE** **********/
/*****************************/

function load_source_datatable(form_element) {

    var formdata = new FormData(form_element);

    var request = new XMLHttpRequest();
    
    request.onreadystatechange = function (response) {
        if (request.readyState === 4) {
            if (request.status === 200) {
                vueApp.addLogEntry(null, "", "Query - Resonse OK");
                vueApp.$data.tableData = JSON.parse(request.responseText)
                vueApp.$data.tableData.tabulatorjs_coldefs = vueApp.$data.tableData.columns.map(function(c) {
                    cdef = {
                                field: c.name,
                                title: c.title, 
                                visible: c.visible,
                                headerTooltip: c.help,
                            };

                    if (c.type == 'float') {
                        // if it is a general float field
                        cdef['formatter'] = function(cell, formatterParams, onRendered) {
                            val = cell.getValue();
                            if (val == null) { 
                                    return ""; 
                            } else {
                                return val.toFixed(3);
                            }
                        }
                    }
                    return cdef;
                });
                make_nice_table();
                vueApp.$data.showDataTable = true;
            } else {
                vueApp.addLogEntry("Error loading data", request.responseText, "Query - Error");
            }
        }
    }

    request.open('POST', '/iop4/api/data/', true);
    request.send(formdata); 
}

// extend tabulator with custom filters
Tabulator.extendModule("filter", "filters", {
    "null":function(filterVal, rowVal){
        return rowVal == null ? true : false;
    },
    "notnull":function(filterVal, rowVal){
        return rowVal != null ? true : false;
    },
    "after":function(filterVal, rowVal){
        return Date.parse(rowVal) > Date.parse(filterVal) ? true : false;
    },
    "before":function(filterVal, rowVal){
        return Date.parse(rowVal) < Date.parse(filterVal) ? true : false;
    }
});

function make_nice_table() {
    
    var table = new Tabulator("#tableDiv", {
        data: vueApp.$data.tableData.data,
        columns: vueApp.$data.tableData.tabulatorjs_coldefs,
        // autoColumns: true,
        layout: "fitDataFill", // "fitDataStretch",
        pagination: true, 
        paginationInitialPage:1, 
        paginationSize: 20,
        paginationSizeSelector: [5, 10, 20, 50],
        paginationCounter: "rows",
        columnDefaults: {
            headerFilter: "input",
            headerMenu: [
                            {
                                label:"Hide Column",
                                action: function(e, column) { column.hide(); }
                            },
                        ]
        },
    });

    // link table controls to this table
    document.getElementById("download-csv").onclick =  function() { table.download("csv", `IOP4_data_${vueApp.$data.tableData.query.source_name}.csv`); };
    document.getElementById("download-pdf").onclick = function() { table.download("pdf", `IOP4_data_${vueApp.$data.tableData.query.source_name}.pdf`, { orientation:"landscape",  title:"title", }); };

    //filters = array of filters currently applied, rows = array of row components that pass the filters
    table.on("dataFiltered", function(filters, rows){ 

        // hack to allow to scroll headers if there are no rows
        if (rows.length == 0)
            document.querySelector('#tableDiv .tabulator-header-contents').style.overflow = 'scroll';
        else 
            document.querySelector('#tableDiv .tabulator-header-contents').style.overflow = 'hidden';

    });

    return table;
}


function show_column_visibility_modal_form() {
    var modal = document.getElementById('column_visibility_modal_form');
    var modal_body = document.querySelector("#column_visibility_modal_form .modal_body");

    modal_body.innerHTML = '';

    var table = Tabulator.findTable('#tableDiv')[0];
    var columns = table.getColumns();

    for(let column of columns){

        //create checkbox element using font awesome icons
        let label = document.createElement("label");

        let span = document.createElement("span");
        span.innerHTML = column.getDefinition().title;
        span.title = column.getDefinition().headerTooltip;

        let checkbox = document.createElement("input");
        checkbox.setAttribute('type', "checkbox");
        checkbox.checked = column.isVisible();

        checkbox.onclick = function() {
                column.toggle();
                checkbox.checked = column.isVisible();
            }

        label.appendChild(checkbox);
        label.appendChild(span);
        modal_body.appendChild(label);
    }

    modal.style.display = 'block';
}

function close_column_visibility_modal_form() {
    var modal = document.getElementById('column_visibility_modal_form');
    modal.style.display = 'none';
}


/**************************/
/********** PLOT **********/
/**************************/


function load_source_plot(form_element) {
    
    document.querySelector('#plotDiv').innerHTML = "";
    document.querySelector('#tablePlotDiv').innerHTML = "";
    document.querySelector('#legendDiv').innerHTML = "";

    var formdata = new FormData(form_element);

    var request = new XMLHttpRequest();
    
    request.onreadystatechange = function (response) {
        if (request.readyState === 4) {
            if (request.status === 200) {
                vueApp.addLogEntry(null, "", "Plot - Resonse OK");
                // embed plot and the legend, 
                plotData = JSON.parse(request.responseText);

                // // for items:
                var fn = function () {
                    elementsIDs = ['plotDiv', 'tablePlotDiv', 'legendDiv'];
                    Object.keys(plotData.render_items[0].roots).forEach((key, index) => {
                        plotData.render_items[0].roots[key] = elementsIDs[index];
                    });
                    Bokeh.embed.embed_items(plotData.doc, plotData.render_items).then((v) => {
                            Bokeh.documents.slice(-1)[0].idle.connect(() => { console.log("document idle, checking layout"); check_plot_layout(); });
                            window.addEventListener('resize', check_plot_layout);
                    });
                }()

                vueApp.$data.showPlot = true;
                vueApp.addLogEntry("Plotted " + plotData.n_points + " points", "Plotted " + plotData.n_points + " points", "Plot - Info");
            } else {
                vueApp.addLogEntry("Error loading plot", request.responseText, "Plot - Error");
                vueApp.$data.showPlot = false;
            }
        }
    }

    request.open('POST', "/iop4/api/plot/", true);
    request.send(formdata);
}


function plot_hide_instrument(e) {

    label = e.getAttribute('data-instrument');

    console.log('Hiding instrument: ' + label)

    if ( !('activeInstruments' in plotData))  {
        plotData.activeInstruments = [];
    }

    if (plotData.activeInstruments.includes(label)) {
        plotData.activeInstruments.splice(plotData.activeInstruments.indexOf(label), 1);
        e.classList.remove('active');
    } else {
        plotData.activeInstruments.push(label);
        e.classList.add('active');
    }

    update_filters();
}

function plot_hide_flag(e) {
    flag = e.getAttribute('data-flag');
    
    console.log('Hiding flag: ' + flag);

    if ( !('activeFlags' in plotData))  {
        plotData.activeFlags = [];
    }

    if (plotData.activeFlags.includes(flag)) {
        plotData.activeFlags.splice(plotData.activeFlags.indexOf(flag), 1);
        e.classList.remove('active');
    } else {
        plotData.activeFlags.push(flag);
        e.classList.add('active');
    }

    update_filters();
}


function update_filters() {

    console.log("Updating filters")

    let invfArray = [];  

    // instruments

    if ('activeInstruments' in plotData) {

        console.log("Building instrument filters")

        plotData.activeInstruments.forEach(label => {
            let gf = new Bokeh.GroupFilter({column_name: 'instrument', group: label});
            let invf = new Bokeh.InversionFilter();
            invf.operand = gf;
            invfArray.push(invf);
        });
    }

    // flags

    if ('activeFlags' in plotData) {

        console.log("Building flag filters")

        plotData.activeFlags.forEach(flag => {
            ff_code = `
                const indices = [];
                
                for (let i = 0; i < source.get_length(); i++){
                    if ( source.data['flags'][i] & (${flag}) ){
                        indices.push(true);
                    } else {
                        indices.push(false);
                    }
                }
                return indices;
            `;
            console.log("ff_code", ff_code);
            let ff = new Bokeh.CustomJSFilter({code:ff_code});

            let invf = new Bokeh.InversionFilter();
            invf.operand = ff;
            invfArray.push(invf);
        });
    }

    // build final filter

    console.log("Building final filter")

    final_filter = new Bokeh.IntersectionFilter()
    final_filter.operands = invfArray 

    Bokeh.documents.slice(-1)[0].get_model_by_name('plot_view').filter = final_filter;

}

function plot_update_errorbars_status() {
    // instead of Bokeh.documents.documents[0] because if we plot several times without refreshing, the documents add up
    if (document.getElementById('cbox_errorbars').checked) {
        plot_show_errorbars();
    } else {
        plot_hide_errorbars();
    } 
}

function plot_hide_errorbars() {
    if (plotData.enable_errorbars) {
        console.log("Hiding errorbars")

        for (var i = 1; i <= 3; i++) {
            // instead of Bokeh.documents.documents[0] becaue if we plot several times without refreshing, the documents add up
            Bokeh.documents.slice(-1)[0].get_model_by_name(`ax${i}_errorbars_renderer`).visible = false;
        }
        
        document.querySelector('#cbox_errorbars').checked = false;
    }
}

function plot_show_errorbars() {
    if (plotData.enable_errorbars) {

        console.log("Showing errorbars")

        for (var i = 1; i <= 3; i++) {
            // instead of Bokeh.documents.documents[0] becaue if we plot several times without refreshing, the documents add up
            Bokeh.documents.slice(-1)[0].get_model_by_name(`ax${i}_errorbars_renderer`).visible = true;
        }

        document.querySelector('#cbox_errorbars').checked = true;
    }
}

function clamp(num, min, max) {
    if (typeof min !== 'number' || isNaN(min)) {
      min = num;
    }
    if (typeof max !== 'number' || isNaN(max)) {
      max = num;
    }
    return num <= min ? min : num >= max ? max : num;
}
  

function get_ymin_ymax(field_y, field_y_err) {
    // get the y_min and y_max columns from the source data
    source = Bokeh.documents.slice(-1)[0].get_model_by_name('source')
    var y = source.data[field_y];
    var y_err = source.data[field_y_err];

    var y_min = new Array(y.length);
    var y_max = new Array(y.length);
    
    for (var i = 0; i < y.length; i++) {
        y_min[i] = y[i] - y_err[i];
        y_max[i] = y[i] + y_err[i];
    }

    return [y_min, y_max];
}

function check_plot_layout() {
    console.log((new Date).toLocaleTimeString(), "Checking plot")
    console.log("Is idle", Bokeh.documents.slice(-1)[0].is_idle)

    /* compute the y_min and y_max columns from the source data if not present */

    source = Bokeh.documents.slice(-1)[0].get_model_by_name('source')

    if (source.data.hasOwnProperty('y1_err') && !source.data.hasOwnProperty('y1_min')) {
        console.log("Computing y1_min and y1_max")
        let [y1_min, y1_max] = get_ymin_ymax('y1', 'y1_err');
        source.data['y1_min'] = y1_min;
        source.data['y1_max'] = y1_max;
    }

    if (source.data.hasOwnProperty('y2_err') && !source.data.hasOwnProperty('y2_min')) {
        console.log("Computing y2_min and y2_max")
        let [y2_min, y2_max] = get_ymin_ymax('y2', 'y2_err');
        source.data['y2_min'] = y2_min;
        source.data['y2_max'] = y2_max;
    }

    if (source.data.hasOwnProperty('y3_err') && !source.data.hasOwnProperty('y3_min')) {
        console.log("Computing y3_min and y3_max")
        let [y3_min, y3_max] = get_ymin_ymax('y3', 'y3_err');
        source.data['y3_min'] = y3_min;
        source.data['y3_max'] = y3_max;
    }

    // source.change.emit();

    /* reorder table and flag gui, we want the flag gui to be as closest as posible to the plot */

    function _pos_at_center(element) {
        const {top, left, width, height} = element.getBoundingClientRect();
        return {
          x: left + width / 2,
          y: top + height / 2
        };
    }
     
    function _distance(a, b) {
       const aPosition = _pos_at_center(a);
       const bPosition = _pos_at_center(b);
     
       return Math.hypot(aPosition.x - bPosition.x, aPosition.y - bPosition.y);  
    }

    plotDiv = document.getElementById("plotDiv");
    tb = document.getElementById("tablePlotDiv");
    flaggui = document.getElementById("FlagGUI");

    if (_distance(plotDiv, flaggui) > _distance(plotDiv, tb)) {
        if (FlagGUI.parentElement.compareDocumentPosition(tb.parentElement.parentElement) & Node.DOCUMENT_POSITION_PRECEDING) {
            FlagGUI.parentElement.after(tb.parentElement.parentElement);
        } else {
            FlagGUI.parentElement.before(tb.parentElement.parentElement);
        }
    }

    // force table to redraw by emmitting the change
    Bokeh.documents.slice(-1)[0]._roots[1].change.emit();

    /* check error bar status */

    console.log("Checking errorbar status")

    plot_update_errorbars_status();

    /* plot */

    if (document.body.clientWidth < 700) {
        Bokeh.documents.slice(-1)[0]._roots[0].toolbar_location = 'above';
        Bokeh.documents.slice(-1)[0]._roots[0].children[0][0].above[0].ticker.desired_num_ticks = 4;
        for (let plot of Bokeh.documents.slice(-1)[0]._roots[0].children) {
            plot[0].left[0].axis_label = "";
            plot[0].title.visible = true;
        }
    } else {
        Bokeh.documents.slice(-1)[0]._roots[0].toolbar_location = 'right';
        Bokeh.documents.slice(-1)[0]._roots[0].children[0][0].above[0].ticker.desired_num_ticks = 5;
        for (let plot of Bokeh.documents.slice(-1)[0]._roots[0].children) {
            plot[0].left[0].axis_label = plot[0].title.text;
            plot[0].title.visible = false;
        }
    }

    // emit all changes

    function _check_plot_emit_changes() {
        // emit changes on the altered elements so the redraw

        for (let plot of Bokeh.documents.slice(-1)[0]._roots[0].children) {
            plot[0].left[0].change.emit();
            plot[0].title.change.emit();
        }

        Bokeh.documents.slice(-1)[0]._roots[0].children[0][0].above[0].change.emit();
    }

    _check_plot_emit_changes();

    // work around because some times the plot takes a bit longer to update
    // there should be a function to just redraw or trigger the resize for
    // but i can not find when its the right time. 200ms should be enough
    // Basically repeats the above code after 200ms.
    
    setTimeout(() => { 
        _check_plot_emit_changes();
    }, 200);

    // alternatively, emit resize event to force the plot to redraw
    // but implement some flag to avoid infinite recursion (not done yet)
    // window.dispatchEvent(new Event('resize'))

    /* plot table size */

    tb_container = document.getElementById("plotTableContainerDiv")
    tb_r = tb_container.parentElement
    tb_r.style.height = 0
    y = 0
    Array.from(tb_r.parentElement.children).forEach( x => { if (x != tb_r) { y += x.offsetHeight + 20; }}); // 20 is the gap

    let maxHeight = parseInt(window.getComputedStyle(tb_r.parentElement).height) - y;

    tb_r.style.height = "";

    if (maxHeight > 100) { // of it is = 0 it will wrapped in column, 100 as error margin
        tb_container.style.maxHeight = maxHeight + 'px';
    } else {
        tb_container.style.maxHeight = "10 em";
    }



    console.log("Plot checked")
}


function set_flag(flag, pts, vals=true) {

    if (pts.length == 0) {
        console.log("No points selected");
        return;
    }

    console.log("Setting flag " + flag + " to " + vals + " for " + pts.length + " items");

    // if val is an integer, make it an array of the same length as pk_array
    if (typeof vals === 'boolean' || typeof vals === 'number') {
        vals = Array(pts.length).fill(vals);
    }

    pk_array = pts.map(x => x['pk']);

    const payload = {
        flag: flag,
        pk_array: pk_array,
        vals: vals,
    };
  
    fetch('/iop4/api/flag/', {
            method: 'POST',
            headers: {
            'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
            credentials: 'same-origin',
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {        
        if (data.success) {
            vueApp.addLogEntry("Flagged " + pts.length + " pts", "Plot - Info");
            update_local_flags(data.new_flag_dict);
        } else {
        console.error('Flagging operation failed.');
        vueApp.addLogEntry("Error fagging", "Plot - Error");
    }
    })
    .catch(error => {
        console.error('Error during flagging operation:', error);
        alert('An error occurred while setting flags.'); // Alert the user to the failure
    });

}

function flags_to_str(flags) {
    let str_arr = [];

    if (flags == 0) {
        str_arr.push("not set");
    } 
          
    if (flags & (1 << 1)) { // bad photometry
        str_arr.push("bad photometry");
    }
          
    if (flags & (1 << 2)) { // bad polarimetry
        str_arr.push("bad polarimetry");
    }

    return str_arr.join(", ");
}

function toggle_flag(flag, pts) {

    if (pts.length == 0) {
        console.log("No points selected");
        return;
    }        

    console.log("Toggling flag " + flag + " for " + pts.length + " items");

    // get the current flag state of the selected items
    // construct the flag array by getting the pt['flags'] for each pt in the table
    old_vals = pts.map(x => Boolean(x['flags'] & flag))
    // toggle the flag
    new_vals = old_vals.map(x => !x)
    // set the new flag state
    set_flag(flag, pts, new_vals);
}

function update_local_flags(new_flag_dict) {

    // update the column data source
    source = Bokeh.documents.slice(-1)[0].get_model_by_name('source')

    source.data['pk'].forEach((pk, index) => {
        if (new_flag_dict.hasOwnProperty(pk)) {
            source.data.flags[index] = new_flag_dict[pk];
        }
    });

    vueApp.$data.selected_plot_idx = vueApp.$data.selected_plot_idx;
    vueApp.$data.selected_refresh++;

    source.change.emit();
}

/*****************************/
/********** Catalog **********/
/*****************************/

function load_catalog() {
    var request = new XMLHttpRequest();
    
    request.onreadystatechange = function (response) {
        if (request.readyState === 4) {
            if (request.status === 200) {
                vueApp.$data.catalog = JSON.parse(request.responseText);
                // map the columns data provided by the api endpoint to the columns data required by the quasar table component
                vueApp.$data.catalog.columns = vueApp.$data.catalog.columns.map(function(c) {
                    cdef = {};
                    cdef = Object.assign(c, c);
                    cdef = Object.assign(cdef, {
                                name: c.name,
                                field: c.name,
                                label: c.title,                                                        
                                align: 'left',
                                style: 'min-width: min-content;',
                    });
                    return cdef;
                });
            } else {
                Quasar.Notify.create("Error loading catalog");
            }
        }
    }

    request.open('GET', "/iop4/api/catalog/", true);
    request.send();
}

/*****************************/
/********** LOGS*** **********/
/*****************************/

function extractTextFromHTML(html) {
    /* This function returns the text content of the given html */
    let parser = new DOMParser();
    let doc = parser.parseFromString(html, 'text/html');
    return doc.body.textContent || "";
}


function highlightTextInHTML(html, re_expr) {
    /* 
        This function looks only inside the text nodes of the given html, and highlights the given pattern by substituting the text with a span with class highlight 
        The re_expr pattern should take care not to break html entities, e.g.
            const re_expr = new RegExp(`(?<!&#?[a-zA-Z]*)${searchstring}(?![a-zA-Z]*;)`, 'gi');
        will make sure to match the search string only if it is not part of an html entity. 
        This is needed because when the value of the text noded is extracted, html entities get 
        transformed to their unicode representation, e.g. &lt; becomes < and &gt; becomes >. Therefore not to break the html these need to be
        transformed back to their html entity representation, e.g. < replaced by &lt; and > replaced by &gt;. Then searching for 'g' would 
        break the entity if it is not excluded from the search.
    */
    let parser = new DOMParser();
    let doc = parser.parseFromString(html, 'text/html');

    function processNode(node) {
        if (node.nodeType === 3) { // Node.TEXT_NODE
            const matches = node.nodeValue.match(re_expr);
            if (matches) {
                const span = document.createElement('span');

                // span.innerHTML = node.nodeValue.replaceAll(re_expr, '<span class="highlight">$&</span>');
                span.innerHTML = node.nodeValue.replace('<','&lt;').replace('>','&gt;').replaceAll(re_expr, '<span class="highlight">$&</span>');

                node.parentNode.replaceChild(span, node);
            }
        } else if (node.nodeType === 1) { // Node.ELEMENT_NODE
            Array.from(node.childNodes).forEach(processNode);
        }
    }

    Array.from(doc.body.childNodes).forEach(processNode);
    return doc.body.innerHTML;
}

async function load_pipeline_log() {

    this.pipeline_log_options.filter_text = ''; // clear the filter text (it would not make sense to push to the filtered array below otherwise)

    console.log(`Loading pipeline log for ${this.log_file}`);

    const log_file = this.log_file;
    const decoder = new TextDecoder('utf-8');
    const response = await fetch(`/iop4/api/log/?log_file=${log_file}`);

    let data = ''
    for await (const chunk of response.body) {
        if (log_file !== this.log_file) {
            console.log("Pipeline log loading cancelled due to new input");
            return; // Exit the loop early
        }

        data += decoder.decode(chunk);
        new_items = data.split('\n').slice(vueApp.$data.pipeline_log.items.length,-1);
        vueApp.$data.pipeline_log.items.push(...new_items);
        // initially put them also in the filtered array so the log appears as they are received
        vueApp.$data.pipeline_log.items_filtered.push(...new_items.filter(is_log_visible)); 
    }

    console.log("Pipeline log loaded!");
}

function is_log_visible(txt) {
    /* whether the given log line should be visible or not */
    
    // show only lines of the selected logging levels
    if ((txt.includes('ERROR')) && (!vueApp.$data.pipeline_log_options.errors)){ return false; }
    if ((txt.includes('WARNING')) && (!vueApp.$data.pipeline_log_options.warnings)){ return false; }
    if ((txt.includes('INFO')) && (!vueApp.$data.pipeline_log_options.info)){ return false; }
    if ((txt.includes('DEBUG')) && (!vueApp.$data.pipeline_log_options.debug)){ return false; }
    if (!(txt.includes('ERROR') || txt.includes('WARNING') || txt.includes('INFO') || txt.includes('DEBUG'))) { return false; }

    // if the filter text is not empty, hide lines that do not contain it
    if (!!(vueApp.$data.pipeline_log_options.filter_text) && (vueApp.$data.pipeline_log_options.filter_text.length > 0)) {
        if (!extractTextFromHTML(txt).toUpperCase().includes(vueApp.$data.pipeline_log_options.filter_text.toUpperCase())) { return false; }
    }

    return true;
}


/* The following code implements the filtering (by is_log_visible) of the pipeline log, and its formatting (by highlightTextInHTML) in a way that
the page does not freeze when the user types in the filter text input. 
It does it by:
    - using a debounce function that waits for the user to stop typing for a given amount of time (debounceDelay) before processing the log items.
    - canceling the processing if a new filtering request has been made. 
    - processing is done in chunks of 100 items, and the next chunk is scheduled to be processed after the current one has finished.
    - using async functions.
This solution is obviously a bit slower but it is much more responsive.
*/

let debounceTimer;
const debounceDelay = 10; // milliseconds
let currentFilterTimestamp = 0;

async function format_pipeline_logs() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        processLogItems();
    }, debounceDelay);
}

async function processLogItems() {

    vueApp.$data.pipeline_log.filter_in_progress = true;

    const startTime = Date.now();
    currentFilterTimestamp = startTime;
    console.log("Filtering pipeline log");

    // Clear the existing array
    vueApp.$data.pipeline_log.items_filtered = [];

    const processChunk = async (start, chunkSize) => {
        for (let i = start; i < start + chunkSize && i < vueApp.$data.pipeline_log.items.length; i++) {
            const txt = vueApp.$data.pipeline_log.items[i];

            // Check if a new filtering request has been made
            if (currentFilterTimestamp !== startTime) {
                console.log("Filtering cancelled due to new input");
                return; // Exit the loop early
            }

            // Check if log is visible
            if (is_log_visible(txt)) {
                let formattedText = txt;

                // Check for highlight condition
                if (vueApp.$data.pipeline_log_options.filter_text) {
                    // const re_expr = new RegExp(vueApp.$data.pipeline_log_options.filter_text, 'gi');
                    const re_expr = new RegExp(`(?<!&#?[a-zA-Z]*)${vueApp.$data.pipeline_log_options.filter_text}(?![a-zA-Z]*;)`, 'gi');
                    formattedText = highlightTextInHTML(txt, re_expr);
                }

                // Push individually
                vueApp.$data.pipeline_log.items_filtered.push(formattedText);
            }
        }

        if (start + chunkSize < vueApp.$data.pipeline_log.items.length) {
            // Schedule next chunk
            setTimeout(() => processChunk(start + chunkSize, chunkSize), 0);
        } else if (currentFilterTimestamp === startTime) {
            console.log("Pipeline log filtered");
            vueApp.$data.pipeline_log.filter_in_progress = false;
        }
    };

    const chunkSize = 100; // Adjust based on performance
    processChunk(0, chunkSize);

}

/*****************************/