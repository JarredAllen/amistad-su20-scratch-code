#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>

// This file contains a bunch of functions which are written in C for
// use in python. For each function written in C, there is a function in
// pure C which executes it, and then another function which wraps it in
// a way that python can interpret.

// Computes the value of $f(theta, M_i)$
double f(double m, double theta) {
    return 1.0 / (1.0 + exp(theta * (m - 0.5)));
}
static PyObject* simulate_f(PyObject* self, PyObject* args) {
    double theta, m;
    if (!PyArg_ParseTuple(args, "dd", &m, &theta)) {
        // The wrong arguments were passed in
        return NULL;
    }
    return Py_BuildValue("d", f(m, theta));
}

// Computes the value of $g_\theta(M_i)$
double g(double m, double theta) {
    return (f(m, theta) - f(0, theta)) / (f(1, theta) - f(0, theta));
}
static PyObject* simulate_g(PyObject* self, PyObject* args) {
    double theta, m;
    if (!PyArg_ParseTuple(args, "dd", &m, &theta)) {
        // The wrong arguments were passed in
        return NULL;
    }
    return Py_BuildValue("d", g(m, theta));
}

// Returns 1 if the agent chooses to affirm, 0 if the agent chooses to
// reject
char get_agent_choice(double m, double theta) {
    return ((double)rand()) / RAND_MAX < g(m, theta);
}
static PyObject* simulate_get_agent_choice(PyObject* self, PyObject* args) {
    double theta, m;
    if (!PyArg_ParseTuple(args, "dd", &m, &theta)) {
        // The wrong arguments were passed in
        return NULL;
    }
    return Py_BuildValue("i", get_agent_choice(m, theta));
}

char* sim_basic(double theta, int agent_count, int initial_yes, int initial_total) {
    int i,
        num_yes = initial_yes,
        num_total = initial_total;
    char response;
    double m;
    char* result = (char*) malloc(sizeof(char)*agent_count);
    if (result == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    for (i=0; i < agent_count; i++) {
        m = (double)num_yes / num_total;
        response = get_agent_choice(m, theta);
        num_yes += response;
        num_total++;
        result[i] = response;
    }
    return result;
}
static PyObject* simulate_sim_basic(PyObject* self, PyObject* args) {
    double theta;
    int agent_count, initial_yes, initial_total;
    if (!PyArg_ParseTuple(args, "diii", &theta, &agent_count, &initial_yes, &initial_total)) {
        // The wrong arguments were passed in
        return NULL;
    }
    return Py_BuildValue("y#", sim_basic(theta, agent_count, initial_yes, initial_total), agent_count);
}

static PyMethodDef SimulateMethods[] = {
    {"f", simulate_f, METH_VARARGS, "Evaluate our \"f\" function"},
    {"g", simulate_g, METH_VARARGS, "Evaluate our \"g\" function"},
    {"get_agent_choice", simulate_get_agent_choice, METH_VARARGS, "Get the choice of the next agent"},
    {"sim_basic", simulate_sim_basic, METH_VARARGS, "Run the basic simulation"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef SimulateModule = {
    PyModuleDef_HEAD_INIT,
    "simulate",
    NULL,
    -1,
    SimulateMethods,
};

PyMODINIT_FUNC PyInit_simulate(void) {
    return PyModule_Create(&SimulateModule);
}
