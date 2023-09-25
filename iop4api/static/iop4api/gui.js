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

async function load_empty_plot(data) {
    const response = await fetch('/iop4api/plot')
    const item = await response.json()
    Bokeh.embed.embed_item(item, "plotDiv")
}

function load_source_plot(form_element) {
    
    document.querySelector('#plotDiv').innerHTML = "";
    document.querySelector('#legendDiv').innerHTML = "";

    var formdata = new FormData(form_element);

    var request = new XMLHttpRequest();
    
    request.onreadystatechange = function (response) {
        if (request.readyState === 4) {
            if (request.status === 200) {
                vueApp.addLogEntry("Plot - Resonse OK", "");
                // embed plot and the legend, 
                plotData = JSON.parse(request.responseText);
                bokeh_plot_promise = Bokeh.embed.embed_item(plotData.item, "plotDiv");
                bokeh_legend_promise = Bokeh.embed.embed_item(plotData.legend, "legendDiv");
                // toggle the errorbars and check the layout when it finishes 
                // bokeh_plot_promise.then((v) => { plot_update_errorbars_status(); check_plot_layout(); }); // needs both items to be loaded since we use .slice(-2) :
                Promise.allSettled([bokeh_plot_promise, bokeh_legend_promise]).then((v) => { plot_update_errorbars_status(); check_plot_layout(); });
                // and add a listener to recheck the layout when the window is resized
                window.addEventListener('resize', check_plot_layout);

                vueApp.$data.showPlot = true;
                vueApp.$data.C2selectedTab = 'plot';
                vueApp.addLogEntry("Plot - Info", "Plotted " + plotData.n_points + " points");
            } else {
                vueApp.addLogEntry("Plot - Error", request.responseText);
                vueApp.$data.C2selectedTab = 'log';
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
                vueApp.addLogEntry("Query - Resonse OK", "");
                make_nice_table(JSON.parse(request.responseText));
                vueApp.$data.C2selectedTab = 'data';
                vueApp.$data.showDataTable = true;
            } else {
                vueApp.addLogEntry("Query - Error", request.responseText);
                vueApp.$data.C2selectedTab = 'log';                
            }
        }
    }

    request.open('POST', '/iop4/api/data/', true);
    request.send(formdata); 
}

function make_nice_table(tableData) {

    var table = new Tabulator("#tableDiv", {
        data: tableData.data,
        columns: tableData.tabulatorjs_columns,
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
    Bokeh.documents.slice(-2)[0].get_model_by_name('plot_view').filter = final_filter;
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
    console.log("Hiding errorbars")

    for (var i = 1; i <= 3; i++) {
        // instead of Bokeh.documents.documents[0] becaue if we plot several times without refreshing, the documents add up
        Bokeh.documents.slice(-2)[0].get_model_by_name(`ax${i}_errorbars_renderer`).visible = false;
    }
    
    document.querySelector('#cbox_errorbars').checked = false;
}

function plot_show_errorbars() {
    console.log("Showing errorbars")

    for (var i = 1; i <= 3; i++) {
        // instead of Bokeh.documents.documents[0] becaue if we plot several times without refreshing, the documents add up
        Bokeh.documents.slice(-2)[0].get_model_by_name(`ax${i}_errorbars_renderer`).visible = true;
    }

    document.querySelector('#cbox_errorbars').checked = true;
}

function check_plot_layout() {
    if (document.body.clientWidth < 700) {
        Bokeh.documents.slice(-2)[0]._roots[0].toolbar_location = 'above';
    } else {
        Bokeh.documents.slice(-2)[0]._roots[0].toolbar_location = 'right';
    }
}

function load_catalog() {
    var request = new XMLHttpRequest();
    
    request.onreadystatechange = function (response) {
        if (request.readyState === 4) {
            if (request.status === 200) {
                vueApp.$data.catalog = JSON.parse(request.responseText);
                vueApp.$data.catalog.columns = Object.keys(vueApp.$data.catalog.data[0]).map((key) => ({
                                                        name: key,
                                                        align: 'left',
                                                        label: key.charAt(0).toUpperCase() + key.slice(1),
                                                        field: key,
                                                        style: 'min-width: min-content;',
                                                    }));
            } else {
                alert("Error loading catalog");
            }
        }
    }

    request.open('GET', "/iop4/api/catalog/", true);
    request.send();
}