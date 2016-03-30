package com.clust4j.except;

import org.junit.Test;

public class TestExcept {

	@Test(expected=IllegalClusterStateException.class)
	public void testICSE1() {
		throw new IllegalClusterStateException();
	}
	
	@Test(expected=IllegalClusterStateException.class)
	public void testICSE2() {
		throw new IllegalClusterStateException("asdf");
	}
	
	@Test(expected=IllegalClusterStateException.class)
	public void testICSE3() {
		throw new IllegalClusterStateException(new Exception());
	}
	
	@Test(expected=IllegalClusterStateException.class)
	public void testICSE4() {
		throw new IllegalClusterStateException("asdf", new Exception());
	}
	
	
	
	@Test(expected=MatrixParseException.class)
	public void testMPE1() {
		throw new MatrixParseException();
	}
	
	@Test(expected=MatrixParseException.class)
	public void testMPE2() {
		throw new MatrixParseException("asdf");
	}
	
	@Test(expected=MatrixParseException.class)
	public void testMPE3() {
		throw new MatrixParseException(new Exception());
	}
	
	@Test(expected=MatrixParseException.class)
	public void testMPE4() {
		throw new MatrixParseException("asdf", new Exception());
	}

	
	
	@Test(expected=ModelNotFitException.class)
	public void testMNFE1() {
		throw new ModelNotFitException();
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testMNFE2() {
		throw new ModelNotFitException("asdf");
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testMNFE3() {
		throw new ModelNotFitException(new Exception());
	}
	
	@Test(expected=ModelNotFitException.class)
	public void testMNFE4() {
		throw new ModelNotFitException("asdf", new Exception());
	}
	
	
	
	@Test(expected=NaNException.class)
	public void testNaN1() {
		throw new NaNException();
	}
	
	@Test(expected=NaNException.class)
	public void testNaN2() {
		throw new NaNException("asdf");
	}
	
	@Test(expected=NaNException.class)
	public void testNaN3() {
		throw new NaNException(new Exception());
	}
	
	@Test(expected=NaNException.class)
	public void testNaN4() {
		throw new NaNException("asdf", new Exception());
	}
	
	
	
	@Test(expected=NonUniformMatrixException.class)
	public void testNUME1() {
		throw new NonUniformMatrixException(1,2);
	}
}
