package com.clust4j;

/**
 * 
 * Models or any {@link com.clust4j.log.Loggable}
 * that should be able to "say their name" should 
 * implement this method.
 * 
 * <p>Other considered names:
 * <p><ul><tt>SelfProfessant</tt></ul>
 * <p><ul><tt>Parrot</tt></ul>
 * <p><ul><tt>EchoChamber</tt></ul>
 * 
 * :-)
 * 
 * <p>
 * @author Taylor G Smith
 */
public interface NamedEntity {
	public String getName();
}
