package com.clust4j.algo;

import java.util.ArrayList;

class ModelSummary extends ArrayList<Object[]> {
	private static final long serialVersionUID = -8584383967988199855L;
	
	ModelSummary(final Object[] ... objs) {
		super();
		for(Object[] o: objs)
			this.add(o);
	}
}
