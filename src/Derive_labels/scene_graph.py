class SceneGraphFormatter:
    def format(self, record):
        nodes = []
        edges = []
        # Emergent concepts as nodes
        for k,v in record.get('emergent_concepts', {}).items():
            nodes.append({'id': k, 'label': k, 'value': v})
        # Contextual hypotheses as attributes
        nodes.append({'id':'context','label':'context_hypotheses','value': record.get('contextual_hypotheses')})
        # Meta probability as attribute
        nodes.append({'id':'meta','label':'meta_probability','value': record.get('meta_prob')})
        # Compositional features as separate concept nodes
        for k,v in record.get('compositional', {}).items():
            nodes.append({'id':k,'label':k,'value':v})
        # Edges: simple fully connected for now
        for i in range(len(nodes)):
            for j in range(i+1,len(nodes)):
                edges.append({'source':nodes[i]['id'],'target':nodes[j]['id'],'relation':'co-occurs'})
        return {'nodes':nodes,'edges':edges}
