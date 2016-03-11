package com.clust4j.algo;

import java.util.concurrent.ConcurrentLinkedDeque;

/**
 * For parallel tasks in models that may need model summaries
 * @author Taylor G Smith
 * @param <T>
 */
abstract class ParallelModelTask<T> extends ParallelTask<T> {
	private static final long serialVersionUID = 2139716909891672022L;
	final ConcurrentLinkedDeque<Object[]> summaries;

	ParallelModelTask(ConcurrentLinkedDeque<Object[]> summaries) {
		super();
		
		this.summaries = summaries;
	}
}
