import networkx as nx
from androguard.misc import AnalyzeAPK
import os

def get_call_graph(dx):
    CG = nx.DiGraph()
    nodes = dx.find_methods('.*', '.*', '.*', '.*')
    for m in nodes:
        API = m.get_method()
        class_name = API.get_class_name()
        method_name = API.get_name()
        descriptor = API.get_descriptor()
        api_call = class_name + '->' + method_name + descriptor
        #api_call = class_name + '->' + method_name

        if len(m.get_xref_to()) == 0:
            continue
        CG.add_node(api_call)

        for other_class, callee, offset in m.get_xref_to():
            _callee = callee.get_class_name() + '->' + callee.get_name() + callee.get_descriptor()
            #_callee = callee.get_class_name() + '->' + callee.get_name()
            CG.add_node(_callee)
            if not CG.has_edge(API, callee):
                CG.add_edge(api_call, _callee)

    return CG

import zipfile
def apk_to_callgraph(apk, existing_files):
    apk_name = apk.split('/')[-1]
    if '.apk' in apk_name:
        apk_name = apk_name.replace('.apk', '')
    else:
        apk_name = apk_name

    #print(apk_name)
    file_name = apk_name + '.txt'
    if file_name in existing_files:
        return None
    elif not zipfile.is_zipfile(apk):
        return None
    elif os.path.getsize(apk) > 10485760:
        return None
    else:
        try:
            a, d, dx = AnalyzeAPK(apk)
            call_graph = get_call_graph(dx=dx)
            cg = call_graph
            return cg

        except:
            return None
