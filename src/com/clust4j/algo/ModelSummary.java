package com.clust4j.algo;

import java.util.ArrayList;

/**
 * The {@link com.clust4j.utils.TableFormatter} uses this class
 * for pretty printing of various models' fit summaries.
 * @author Taylor G Smith
 */
class ModelSummary extends ArrayList<Object[]> {
	private static final long serialVersionUID = -8584383967988199855L;
	
	ModelSummary(final Object[] ... objs) {
		super();
		for(Object[] o: objs)
			this.add(o);
	}
}
