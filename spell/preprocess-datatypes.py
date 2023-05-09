from lxml import etree

namespaces = {
    "owl": "http://www.w3.org/2002/07/owl#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
}


def expand_namespace(namespace, item):
    return "{%s}%s" % (namespaces.get(namespace), item)


tag_onto = expand_namespace("owl", "Ontology")
tag_ni = expand_namespace("owl", "NamedIndividual")
tag_type = expand_namespace("rdf", "type")
tag_class = expand_namespace("owl", "Class")
tag_thing = expand_namespace("owl", "Thing")
tag_object_prop = expand_namespace("owl", "ObjectProperty")
tag_data_prop = expand_namespace("owl", "DatatypeProperty")
tag_annotation_prop = expand_namespace("owl", "AnnotationProperty")
attr_resource = expand_namespace("rdf", "resource")
attr_about = expand_namespace("rdf", "about")
attr_datatype = expand_namespace("rdf", "datatype")
tag_range = expand_namespace("rdfs", "range")


def addCN(root, name):
    cn = etree.Element(tag_class)
    cn.attrib[attr_about] = root.nsmap[None] + name
    root.insert(1, cn)
    return root.nsmap[None] + name


def process_owl(file, outfile):
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(file, parser)

    treeroot = tree.getroot()

    datatype_props = {}
    prop_values = {}
    classes = set()

    prefix = treeroot.nsmap[None]

    for element in treeroot:
        if element.tag == tag_class:
            classes.add(element.attrib[attr_about])
        elif element.tag == tag_data_prop:
            for child in element:
                if child.tag == tag_range:
                    name = element.attrib[attr_about]
                    for ns in treeroot.nsmap.values():
                        if ns in name:
                            name = name[len(ns) :]
                    datatype_props[name] = child.attrib[attr_resource]
                    prop_values[name] = set()
        elif (
            element.tag == tag_ni or element.tag == tag_thing or element.tag in classes
        ):
            for child in element:
                tag = etree.QName(child.tag).localname
                if tag in datatype_props.keys():
                    prop_values[tag].add(child.text)

    print(datatype_props)
    EQ = {}
    NEQ = {}
    GEQ = {}
    LEQ = {}
    for prop in datatype_props.keys():
        GEQ[prop] = {}
        LEQ[prop] = {}
        NEQ[prop] = {}
        EQ[prop] = {}
        if (
            "#double" in datatype_props[prop]
            or "#int" in datatype_props[prop]
            or "#nonNegativeInteger" in datatype_props[prop]
        ):
            for val in prop_values[prop]:
                safeval = val.replace(".", "p")
                GEQ[prop][val] = addCN(treeroot, "{}geq{}".format(prop, safeval))
                LEQ[prop][val] = addCN(treeroot, "{}leq{}".format(prop, safeval))
        if "#string" in datatype_props[prop]:
            for val in prop_values[prop]:
                safeval = val.replace(" ", "_")
                EQ[prop][val] = addCN(treeroot, "{}is{}".format(prop, safeval))
                NEQ[prop][val] = addCN(treeroot, "{}isNot{}".format(prop, safeval))
        if "#boolean" in datatype_props[prop]:
            EQ[prop]["true"] = addCN(treeroot, "{}isTrue".format(prop))
            NEQ[prop]["true"] = addCN(treeroot, "{}isFalse".format(prop))

    for element in treeroot:
        if element.tag == tag_ni or element.tag == tag_thing or element.tag in classes:
            toadd = set()
            for child in element:
                tag = etree.QName(child.tag).localname
                if tag not in datatype_props:
                    continue
                if (
                    "#double" in datatype_props[tag]
                    or "#int" in datatype_props[tag]
                    or "#nonNegativeInteger" in datatype_props[tag]
                ):
                    val = float(child.text)
                    for v2 in prop_values[tag]:
                        if val >= float(v2):
                            toadd.add(GEQ[tag][v2])
                        if val <= float(v2):
                            toadd.add(LEQ[tag][v2])
                if "#string" in datatype_props[tag]:
                    for v2 in prop_values[tag]:
                        if child.text == v2:
                            toadd.add(EQ[tag][v2])
                        else:
                            toadd.add(NEQ[tag][v2])
                if "#boolean" in datatype_props[tag]:
                    if child.text == "true":
                        toadd.add(EQ[tag]["true"])
                    else:
                        toadd.add(NEQ[tag]["true"])

            for tp in toadd:
                sb = etree.SubElement(element, tag_type)
                sb.attrib[attr_resource] = tp

    with open(outfile, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True, pretty_print=True)

    # for (_, elem) in etree.iterparse(file, events=("end",), remove_blank_text=True):
    #     # We are only interested in top-level statements
    #     if elem.getparent() != None and elem.getparent().getparent() != None:
    #         continue

    #     if elem.tag == tag_class:
    #         pass
    #     elif elem.tag == tag_object_prop:
    #         pass
    #     elif elem.tag == tag_data_prop:
    #         print(etree.tostring(elem, pretty_print = True).decode())
    #         pass
    #     elif elem.tag == tag_annotation_prop:
    #         pass
    #     elif (
    #         elem.tag == tag_ni
    #         or elem.tag == tag_thing
    #         or (attr_about in elem.attrib and elem.tag != tag_onto)
    #     ):
    #         pass

    return


def main():
    process_owl("nctrer.owl", "nctrer-processed.owl")


if __name__ == "__main__":
    # execute only if run as a script
    main()
