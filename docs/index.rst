.. _emg3d-manual:

###################
emg3d Documentation
###################

:Release: |version|
:Date: |today|
:Source: `github.com/emsig/emg3d <https://github.com/emsig/emg3d>`_

----

.. toctree::
   :hidden:

   manual/index
   api/index
   dev/index


.. panels::

    ---

    :fa:`book,fa-4x`

    User Guide
    ^^^^^^^^^^

    The manual contains installation instructions, information on how to get
    started, tips and tricks, background theory, how to cite emg3d, important
    references, license information, and much more!

    +++

    .. link-button:: manual
            :type: ref
            :text: To the user guide
            :classes: btn-block btn-info stretched-link

    ---

    :fa:`images,fa-4x`

    Gallery
    ^^^^^^^

    The gallery contains examples and tutorials on the usage of emg3d, and is
    generally the best way to get started. Download them and modify them to
    your needs! It is also a good place to see what is possible with emg3d.

    +++

    .. link-button:: https://emsig.xyz/emg3d-gallery
            :type: url
            :text: To the gallery
            :classes: btn-block btn-info stretched-link

    ---

    :fa:`code,fa-4x`

    API reference
    ^^^^^^^^^^^^^

    The API reference of emg3d is extensive, and includes almost every function
    and class. A lot of the underlying theory is described in the docstring of
    the functions, particularly in the ``core`` module.

    +++

    .. link-button:: api
            :type: ref
            :text: To the API reference
            :classes: btn-block btn-info stretched-link

    ---

    :fa:`terminal,fa-4x`

    Developer guide
    ^^^^^^^^^^^^^^^

    emg3d is a community code, please help to shape its future! From typos to
    bugs to new developments, every contribution is welcomed. This section is
    the best way to get started.

    +++

    .. link-button:: development
            :type: ref
            :text: To the development guide
            :classes: btn-block btn-info stretched-link

    ---
    :column: col-lg-12 p-2

    :fa:`comments,fa-4x`

    Contact
    ^^^^^^^

    .. link-button:: https://emsig.xyz/#contributing-contact
            :type: url
            :text: Go to «Contributing & Contact» @ emsig.xyz
            :classes: btn-block btn-info stretched-link
    ---
    :column: col-lg-12 p-2

    .. dropdown:: About the name and logo of **emg3d**

        The name **emg3d** is a portmanteau of *electromagnetic* (em),
        *multigrid* (mg), and *three-dimensional* (3D): emg3d is a
        three-dimensional multigrid solver for electromagnetic diffusion.

        The symbol stands for the full ('F') multigrid cycling scheme through
        the different grid sizes. This is given, for a maximum of three
        coarsening steps, by::

            h_
           2h_ \          /
           4h_  \    /\  /
           8h_   \/\/  \/


        See, for example, the theory section around :numref:`Figure %s
        <Muld06_Fig5>`.

        .. image:: ./_static/emg3d-logo.svg

