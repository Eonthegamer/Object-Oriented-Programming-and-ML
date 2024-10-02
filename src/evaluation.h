#ifndef EVALUATION_H
#define EVALUATION_H

#include "expression.h"
#include "tensor.h"

class evaluation
{
public:
    evaluation(const std::vector<expression> &exprs);

    void add_kwargs_double(
        const char *key,
        double value);

    void add_kwargs_ndarray(
        const char *key,
        int dim,
        size_t shape[],
        double data[]);

    // return 0 for success
    int execute();

    // return the variable computed by the last expression
    //double &get_result();

    void set_result(tensor valor);
    tensor &get_result();

private:
    tensor result_;
    const std::vector<expression> exprs_;
    std::map<std::string, tensor> variables_;
    std::map<std::string, double> kwargs_;
    //std::map<int, double> values;
}; // class evaluation

#endif // EVALUATION_H
