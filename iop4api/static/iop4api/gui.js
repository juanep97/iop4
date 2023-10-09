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
                         plot_update_errorbars_status(); 
                         check_plot_layout();
                         window.addEventListener('resize', check_plot_layout);
                        });
                }()

                // // for json_items:
                // Bokeh.embed.embed_item(plotData.plot.div_plot, "plotDiv");
                // bokeh_plot_promise = Bokeh.embed.embed_item(plotData.plot, "plotDiv");
                // bokeh_legend_promise = Bokeh.embed.embed_item(plotData.legend, "legendDiv");
                // bokeh_table_promise = Bokeh.embed.embed_item(plotData.table, "tablePlotDiv");
                // // toggle the errorbars and check the layout when it finishes 
                // Promise.allSettled([bokeh_plot_promise, bokeh_legend_promise, bokeh_table_promise]).then((v) => { plot_update_errorbars_status(); check_plot_layout(); });
                // // and add a listener to recheck the layout when the window is resized
                // window.addEventListener('resize', check_plot_layout);

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


function load_source_datatable(form_element) {

    var formdata = new FormData(form_element);

    var request = new XMLHttpRequest();
    
    request.onreadystatechange = function (response) {
        if (request.readyState === 4) {
            if (request.status === 200) {
                vueApp.addLogEntry(null, "", "Query - Resonse OK");
                make_nice_table(JSON.parse(request.responseText));
                vueApp.$data.showDataTable = true;
            } else {
                vueApp.addLogEntry("Error loading data", request.responseText, "Query - Error");
            }
        }
    }

    request.open('POST', '/iop4/api/data/', true);
    request.send(formdata); 
}

function make_nice_table(tableData) {
    
    var table = new Tabulator("#tableDiv", {
        data: tableData.data,
        columns: tableData.columns,
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
    document.getElementById("download-csv").onclick =  function() { table.download("csv", "data.csv"); };
    document.getElementById("download-pdf").onclick = function() { table.download("pdf", "data.pdf", { orientation:"landscape",  title:"title", }); };

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

    
/* column visibility form */

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


function plot_hide_instrument(e) {

    label = e.getAttribute('data-instrument');

    console.log('Hiding instument: ' + label)

    if ( !('activeFilters' in plotData))  {
        plotData.activeFilters = [];
    }

    if (plotData.activeFilters.includes(label)) {
        plotData.activeFilters.splice(plotData.activeFilters.indexOf(label), 1);
        e.classList.remove('active');
    } else {
        plotData.activeFilters.push(label);
        e.classList.add('active');
    }

    let invfArray = [];  

    plotData.activeFilters.forEach(label => {
        let gf = new Bokeh.GroupFilter({column_name: 'instrument', group: label});
        let invf = new Bokeh.InversionFilter();
        invf.operand = gf;
        invfArray.push(invf);
    });

    final_filter = new Bokeh.IntersectionFilter()
    final_filter.operands = invfArray 

    // instead of Bokeh.documents.documents[0] becaue if we plot several times without refreshing, the documents add up
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

function check_plot_layout() {
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
    for (let plot of Bokeh.documents.slice(-1)[0]._roots[0].children) {
        plot[0].left[0].change.emit();
        plot[0].title.change.emit();
    }

    Bokeh.documents.slice(-1)[0]._roots[0].children[0][0].above[0].change.emit();
    Bokeh.documents.slice(-1)[0]._roots[0].children[0][0].change.emit();
}

function load_catalog() {
    var request = new XMLHttpRequest();
    
    request.onreadystatechange = function (response) {
        if (request.readyState === 4) {
            if (request.status === 200) {
                vueApp.$data.catalog = JSON.parse(request.responseText);
                vueApp.$data.catalog.columns = vueApp.$data.catalog.columns.map((c) => ({
                                                        name: c.name,
                                                        align: 'left',
                                                        label: c.title,
                                                        field: c.field,
                                                        style: 'min-width: min-content;',
                                                    }));
            } else {
                Quasar.Notify.create("Error loading catalog");
            }
        }
    }

    request.open('GET', "/iop4/api/catalog/", true);
    request.send();
}


function load_pipeline_log() {
    const decoder = new TextDecoder('utf-8');
    
    let output = '';

    fetch('/iop4/api/log/')
        .then(response => {
            vueApp.$data.pipeline_log.isLoaded = false;
            const reader = response.body.getReader();
            return new ReadableStream({
                start(controller) {
                function push() {
                    reader.read().then(({ done, value }) => {
                        if (done) {
                            controller.close();
                            vueApp.$data.pipeline_log.isLoaded = true;
                            vueApp.$data.pipeline_log.data = output;
                            show_pipeline_log();
                            return;
                        }
                        output += decoder.decode(value);
                        push();
                    });
                }
                push();
            }
        });
    });
}

function extractTextFromHTML(html) {
    let parser = new DOMParser();
    let doc = parser.parseFromString(html, 'text/html');
    return doc.body.textContent || "";
}

highlight_parser = new DOMParser();

function highlightTextInHTML(html, re_expr) {
    let doc = highlight_parser.parseFromString(html, 'text/html');

    //if (re_expr.test(doc.textContent)) {
        doc.body.querySelectorAll('*').forEach(el => {
            // Check if it's a innermost text node
            if (el.childNodes.length === 1 && el.firstChild.nodeType === 3) { 
                el.innerHTML = el.innerHTML.replaceAll(re_expr, function(match) {
                    return `<span class="highlight">${match}</span>`;
                });
            }
        });
    //}

    return doc.body.innerHTML;
}

function highlightTextInHTML2(html, re_expr) {
    return html.replaceAll(re_expr, function(match) {
        return `<span class="highlight">${match}</span>`;
    });
}

function show_pipeline_log() {
    vueApp.$data.pipeline_log.items = vueApp.$data.pipeline_log.data.split('\n').filter((txt) => {
        // if the filter text is not empty, hide lines that do not contain it
        if ((vueApp.$data.pipeline_log_options.filter_text != null) && (vueApp.$data.pipeline_log_options.filter_text != '') && (vueApp.$data.pipeline_log_options.filter_text.length > 2)) {
            if (!extractTextFromHTML(txt).toUpperCase().includes(vueApp.$data.pipeline_log_options.filter_text.toUpperCase())) { return false; }
        }
        // show only lines of the selected logging levels
        if ((txt.includes('ERROR')) && (vueApp.$data.pipeline_log_options.errors)){ return true; }
        if ((txt.includes('WARNING')) && (vueApp.$data.pipeline_log_options.warnings)){ return true; }
        if ((txt.includes('INFO')) && (vueApp.$data.pipeline_log_options.info)){ return true; }
        if ((txt.includes('DEBUG')) && (vueApp.$data.pipeline_log_options.debug)){ return true; }
        return false
    });
    
    // if the filter text is not empty, highlight the text
    if ((vueApp.$data.pipeline_log_options.filter_text != null) && (vueApp.$data.pipeline_log_options.filter_text != '' && vueApp.$data.pipeline_log_options.filter_text.length > 2)) {
        re_expr = new RegExp(vueApp.$data.pipeline_log_options.filter_text, 'gi');
        for (let i = 0; i < vueApp.$data.pipeline_log.items.length; i++) {
            vueApp.$data.pipeline_log.items[i] = highlightTextInHTML2(vueApp.$data.pipeline_log.items[i], re_expr);
        }
    }
}