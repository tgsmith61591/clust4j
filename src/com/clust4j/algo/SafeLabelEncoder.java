package com.clust4j.algo;

/**
 * A type of {@link LabelEncoder} that will allow a single class
 * @author Taylor G Smith
 */
public class SafeLabelEncoder extends LabelEncoder {
	private static final long serialVersionUID = -7128029823397014669L;

	public SafeLabelEncoder(int[] labels) {
		super(labels);
	}
	
	@Override
	protected boolean allowSingleClass() {
		return true;
	}
	
	@Override
	public SafeLabelEncoder fit() {
		synchronized(this) {
			return (SafeLabelEncoder) super.fit();
		}
	}
}
