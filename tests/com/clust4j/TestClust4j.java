package com.clust4j;

import java.io.IOException;

import org.junit.Test;

public class TestClust4j {
	final static Clust4j c4j = new Clust4j(){
		private static final long serialVersionUID = 1L;
	};

	@Test(expected=NullPointerException.class)
	public void testSaveNPE() throws IOException {
		c4j.saveObject(null);
	}

	@Test(expected=NullPointerException.class)
	public void testLoadNPE() throws IOException, ClassNotFoundException {
		Clust4j.loadObject(null);
	}
}
