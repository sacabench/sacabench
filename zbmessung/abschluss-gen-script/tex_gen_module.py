def latex_rotate(s):
    return "\\rotatebox[origin=c]{{90}}{{{}}}".format(s)

def latex_color(s, cname):
    return "{{\\color{{{}}}{}}}".format(cname, s)

def latex_colorbox(s, cname):
    return "\\colorbox{{{}}}{{{}}}".format(cname, s)

def nice_file(f, rotate=True):
    f = f.replace(".200MB", "")
    f = f.replace("_", "\\_")
    f = "\\texttt{{{}}}".format(f)
    if rotate:
        f = latex_rotate(f)
    return f

def nice_algoname(n):
    def fix_algo_name(n):
        if n == "DSS":
            n = "DivSufSort"
        n = n[0].capitalize() + n[1:]
        return n

    n = fix_algo_name(n)
    n = "\\text{{{}}}".format(n)
    #n = n.replace("_par_ref", "}äää{\\text{par,ref}}{")
    n = n.replace("_ref", "}äää{\\text{ref}}{")
    #n = n.replace("_par", "}äää{\\text{par}}{")
    n = n.replace("_", "\\_")
    n = n.replace("äää", "_")
    n = n.replace("{}", "")
    n = "${}$".format(n)
    return n

def expanded_headings_size(headings):
    total_width = 1
    for e in headings:
        total_width *= len(e)
    return total_width

def expand_x_headings(x_headings):
    total_width = expanded_headings_size(x_headings)
    rv = []
    for e in x_headings:
        rv.append([])

    v = rv[0]
    for e in x_headings[0]:
        v.append((e, int(total_width / len(x_headings[0]))))
        if len(x_headings) > 1:
            rrv = expand_x_headings(x_headings[1:])[0]
            for i,sublist in enumerate(rrv):
                rv[i+1] += sublist

    return (rv, total_width)

class tex_figure_wrapper:
    tex = ""
    def __init__(self, tex):
        self.tex = tex
    def wrap_resize_box(self):
        out = ""
        out += "\\resizebox{\\textwidth}{!}{\n"
        out += self.tex
        out += "}\n"
        self.tex = out
        return self
    def wrap_table(self, title, label, legend):
        out = ""
        #out += "\\subsection{{{}}}\n".format(title)
        #out += "\n{}\n".format(header_text).replace("%LABEL", label)
        out += "\\begin{table}[h]\n"
        out += self.tex
        out += "\\caption{{{}}}\n".format(title)
        out += "\\label{{{}}}\n".format(label)
        if legend:
            out += (legend + "\n")
        out += "\\end{table}\n"
        self.tex = out
        return self
    def wrap_centering(self):
        self.tex = "\\centering\n{}".format(self.tex)
        return self

def generate_latex_table_single_2(multidim_array,
                                  x_headings,
                                  y_headings,
                                  cfg,
                                  val_align):
    assert len(x_headings) >= 1
    assert len(y_headings) == 1
    x_omit = set()
    for c in cfg:
        if c[0] == "x":
            x_omit.add(c[1])

    rv = expand_x_headings(x_headings)
    x_cells = rv[1]
    x_cell_levels = rv[0]

    tex = ""
    tex += "\\begin{tabular}{l" + (val_align * x_cells) + "}\n"
    tex += "\\toprule\n"

    for (x_cell_level_depth, x_cell_level) in enumerate(x_cell_levels):
        x_cell_level_fmt = []
        for (x_cell, span) in x_cell_level:
            if span == 1:
                x_cell_level_fmt.append(x_cell)
            else:
                x_cell_level_fmt.append("\\multicolumn{{{}}}{{c}}{{{}}}".format(span,x_cell))

        if not x_cell_level_depth in x_omit:
            tex += "     & {} \\\\\n".format(" & ".join(x_cell_level_fmt))

    tex += "\\midrule\n"

    for (y_heading, dataline) in zip(y_headings[0], multidim_array):
        tex += "    {} & {} \\\\\n".format(y_heading, " & ".join(dataline))

    tex += "\\bottomrule\n"
    tex += "\\end{tabular}\n"

    return tex_figure_wrapper(tex)


def generate_latex_table(outer_matrix, threads_and_sizes, algorithms, files):
    table_cfg_kinds = [
        {
            "kind": "sa_check",
            "title": "\\sa Korrektheit Weak-Scaling Large",
            "text": """
Wir Überprüfen zunächst, ob alle Testdaten von allen Implementierungen
korrekt verarbeitet werden konnten. Die Messergebnisse enthalten hierfür von \\texttt{-{}-check} erzeugte Informationen und können in \\cref{%LABEL} eingesehen werden.
            """[1:-1],
            "label": "messung:tab:sa-chk-weak-large",
        },
        {
            "kind": "time",
            "title": "Laufzeit Weak-Scaling Large in Minuten",
            "text": """
Wir betrachten nun in \\cref{%LABEL} die mediane Laufzeit, die jeder Algorithmus für die jeweiligen Testdaten erreicht hat.
Pro Eingabe sind jeweils die besten drei Algorithmen mit Grün markiert, und die schlechtesten drei mit Rot.
            """[1:-1],
            "label": "messung:tab:duration-weak-large",
        },
        {
            "kind": "mem",
            "title": "Speicherpeak Weak-Scaling Large in GiB",
            "text": """
\\Cref{%LABEL} entnehmen wir den mediane Speicherverbrauch. Der Wert setzt sich zusammen aus der Allokation für den Eingabetext inklusive der vom Algorithmus benötigten extra Sentinel Bytes, die Allokation für das Ausgabe \sa und dem vom Algorithmus selbst benötigten Speichers.
Pro Eingabe sind erneut die besten drei Algorithmen mit Grün markiert, und die schlechtesten drei mit Rot.
            """[1:-1],
            "label": "messung:tab:mem-weak-large",
        },
    ]

    for cfg in table_cfg_kinds:
        e = generate_latex_table_list(cfg, outer_matrix, threads_and_sizes, algorithms, files)
        print(e)

def generate_latex_table_list(cfg, outer_matrix, threads_and_sizes, algorithms, files):
    check_log = cfg["logdata"]

    def tex_sa_check(key, which):
        nonlocal outer_matrix
        nonlocal check_log

        def ret(ts, f, algorithm_name):
            nonlocal outer_matrix
            nonlocal which
            nonlocal key

            if outer_matrix[ts][f][algorithm_name]["data"] != "exists":
                k = (algorithm_name, f, str(ts[0]))
                if check_log.get(k) == "timeout":
                    return "{\color{orange}\\faClockO}"
                if check_log.get(k) == "crash":
                    return "{\color{violet}\\faBolt}"
                if check_log.get(k) == "oom":
                    return "{\color{purple}\\faFloppyO}"

                return "{\color{darkgray}--}"

            if outer_matrix[ts][f][algorithm_name][which][key] == "ok":
                return "\\cmarkc" #"{\color{green}\\faCheck}"
            else:
                return "\\xmarkc" #"{\color{red}\\faTimes}"

        return ret

    def tex_number(key, fmt, which):
        nonlocal outer_matrix
        nonlocal algorithms

        def ret(ts, f, algorithm_name):
            nonlocal outer_matrix
            nonlocal fmt
            nonlocal which
            nonlocal key
            nonlocal algorithms


            if outer_matrix[ts][f][algorithm_name]["data"] != "exists":
                return "{\color{darkgray}--}"

            if outer_matrix[ts][f][algorithm_name][which]["check_result"] != "ok":
                return "{\color{darkgray}--}"

            datapoints = set()
            for ai in algorithms:
                if not outer_matrix[ts][f][ai]["data"] != "exists":
                    if not outer_matrix[ts][f][ai][which]["check_result"] != "ok":
                        d = outer_matrix[ts][f][ai][which][key]
                        datapoints.add((d, ai))

            datapoints = list(sorted(datapoints, key = lambda t: t[0]))
            datapoints = [(d, n, i) for (i, (d, n)) in enumerate(datapoints)]
            size = len(datapoints)

            (d, i) = next((d, i) for (d, n, i) in datapoints if n == algorithm_name)

            #print((d, i))
            #print()

            raw = outer_matrix[ts][f][algorithm_name][which][key]
            assert raw == d

            uniq_datapoints = list(sorted(set(map(lambda dx: dx[0], datapoints))))

            classif = 0
            if len(uniq_datapoints) >= 3:
                third_best = uniq_datapoints[2]
                third_worst = uniq_datapoints[-3]
                if d >= third_worst:
                    classif = -1
                if d <= third_best:
                    classif = 1
            else:
                classif = 1

            formated = fmt(d, classif)

            return formated

        return ret

    def time_fmt(d, classif):
        d = d / 1000
        d = d / 60
        d = "{:0.2f}".format(d)
        #d = latex_rotate("\\ " + d + "\\ ")

        if classif > 0:
            d = latex_color(d, "green!60!black")
        elif classif < 0:
            d = latex_color(d, "red")

        return d

    def mem_fmt(d, classif):
        d = d / 1024
        d = d / 1024
        d = d / 1024
        d = "{:0.3f}".format(d)
        #d = latex_rotate("\\ " + d + "\\ ")

        if classif > 0:
            d = latex_color(d, "green!60!black")
        elif classif < 0:
            d = latex_color(d, "red")

        return d

    legend = cfg["legend"]
    title = cfg["title"]
    get_data = {
        "sa_check": tex_sa_check("check_result", "med"),
        "time": tex_number("duration", time_fmt, "med"),
        "mem": tex_number("mem_local_peak", mem_fmt, "med"),
    }[cfg["kind"]]
    val_align = {
        "sa_check": "c",
        "time": "r",
        "mem": "r",
    }[cfg["kind"]]
    label = cfg["label"]
    omit_headings = cfg["omit_headings"]
    single_cfg = omit_headings

    # data, algorithms, files
    x_tex_headings = [
        [nice_file(s) for s in files],
        ["{}".format(t) for (t, s) in threads_and_sizes],
    ]
    y_tex_headings = [
        [nice_algoname(algorithm_name) for algorithm_name in algorithms],
    ]

    x_headings = [
        files,
        threads_and_sizes,
    ]
    y_headings = [
        algorithms,
    ]

    def rec(headings):
        if len(headings) == 1:
            return [[e] for e in headings[0]]
        else:
            r = []
            for e in headings[0]:
                ee = rec(headings[1:])
                for eee in ee:
                    eee.insert(0, e)
                r += ee
            return r

    multidim_array = []
    for [algorithm] in rec(y_headings):
        multidim_array.append([])
        for [file, threads_and_size] in rec(x_headings):
            #cell = "{}-{}-{}".format(nice_algoname(algorithm), nice_file(file, False), threads_and_size[0])
            cell = get_data(threads_and_size, file, algorithm)
            multidim_array[-1].append(cell)
            #print(algorithm, file, threads_and_size)

    #pprint.pprint(multidim_array)

    out = generate_latex_table_single_2(multidim_array,
                                        x_tex_headings,
                                        y_tex_headings,
                                        single_cfg,
                                        val_align)
    tex_fragment = out.wrap_resize_box().wrap_centering().wrap_table(title, label, legend)

    return tex_fragment.tex
