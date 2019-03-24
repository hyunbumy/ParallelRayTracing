CC=gcc
CXX=g++
RM=rm -f
CPPFLAGS=-g -O3 -Wall -c -MMD -MP
LDFLAGS=-g
LDLIBS=

BUILDDIR=build
TESTDIR=test
SRCS=$(wildcard *.cpp)
DEPS=$(OBJS:.o=.d)
OBJS=$(patsubst %.cpp,$(BUILDDIR)/%.o,$(SRCS))

all: dir $(BUILDDIR)/main

dir:
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/main: $(OBJS)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(BUILDDIR)/%.o: %.cpp
	$(CXX) $(CPPFLAGS) $< -o $@

clean:
	rm $(BUILDDIR)/*

-include $(DEPS)
